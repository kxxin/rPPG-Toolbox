"""Training-only temporal extension of an existing, immutable cached split.

No resampling, padding, filtering or re-normalization is performed here.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader


class ConsecutiveClipDataset(Dataset):
    def __init__(self, base, clip_length=160, sequence_length=640):
        if sequence_length <= clip_length or sequence_length % clip_length:
            raise ValueError('Sequence length must be a larger integer multiple of cached clip length')
        if base.data_format != 'NCDHW':
            raise ValueError('ConsecutiveClipDataset requires NCDHW')
        self.base = base
        self.inputs = list(base.inputs)  # Original CSV membership for the trainer split audit.
        self.data_format = base.data_format
        self.clip_length, self.sequence_length = clip_length, sequence_length
        self.factor = sequence_length // clip_length
        subjects = defaultdict(list)
        parents = defaultdict(set)
        for index, path in enumerate(self.inputs):
            portable = str(path).replace('\\', '/')
            match = re.fullmatch(r'(subject\d+)_input(\d+)\.npy', portable.rsplit('/', 1)[-1])
            if match is None:
                raise ValueError('Expected UBFC subjectN_inputK.npy: ' + str(path))
            subject, chunk = match.group(1), int(match.group(2))
            subjects[subject].append((chunk, index))
            parents[subject].add(portable.rsplit('/', 1)[0])
        if not subjects:
            raise ValueError('Empty training split')
        self.groups, self.group_subjects = [], []
        for subject in sorted(subjects):
            pairs = sorted(subjects[subject])  # Numeric, never lexicographic chunk order.
            ids = [p[0] for p in pairs]
            if len(parents[subject]) != 1 or ids != list(range(len(ids))):
                raise ValueError('Duplicate, missing or nonzero-start chunks / mixed caches for ' + subject)
            if len(ids) < self.factor:
                raise ValueError('Cannot form a full sequence without padding for ' + subject)
            starts = list(range(0, len(ids) - self.factor + 1, self.factor))
            # Include the real tail. Overlap is between groups, never inside a sequence.
            if starts[-1] != len(ids) - self.factor:
                starts.append(len(ids) - self.factor)
            for start in starts:
                self.groups.append(tuple(p[1] for p in pairs[start:start+self.factor]))
                self.group_subjects.append(subject)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        items = [self.base[i] for i in self.groups[index]]
        for data, label, *_ in items:
            if data.ndim != 4 or data.shape[1] != self.clip_length or label.shape != (self.clip_length,):
                raise ValueError('Cached input/label length differs from configured clip length')
        video = np.concatenate([item[0] for item in items], axis=1)
        label = np.concatenate([item[1] for item in items], axis=0)
        return video, label, self.group_subjects[index], str(index)


class ExposureMatchedSampler(Sampler):
    """Select near-original frame exposure, rotating least-used groups across epochs.

Every subject occurs each epoch. No group repeats within an epoch. Tail groups
may share cached clips. Plans use an isolated RNG and never inspect labels.
"""
    def __init__(self, dataset, epochs, seed=100):
        self.epoch = 0
        self.count = int(math.floor(len(dataset.inputs) / dataset.factor + 0.5))
        subjects = sorted(set(dataset.group_subjects))
        if not len(subjects) <= self.count <= len(dataset):
            raise ValueError('Cannot match exposure while retaining every subject each epoch')
        by_subject = {s: [i for i, v in enumerate(dataset.group_subjects) if v == s] for s in subjects}
        subject_sizes = {s: len({i for g in by_subject[s] for i in dataset.groups[g]}) for s in subjects}
        subject_used = Counter()
        used = np.zeros(len(dataset), dtype=np.int64)
        rng = np.random.RandomState(seed)
        self.plans = []
        for epoch in range(epochs):
            tie = rng.random_sample(len(dataset))
            selected = set()
            current = Counter()

            def add_subject(subject):
                available = set(by_subject[subject]) - selected
                chosen = min(available, key=lambda i: (used[i], tie[i]))
                selected.add(chosen)
                current[subject] += 1

            for subject in subjects:
                add_subject(subject)
            # Preserve subjects' relative frame exposure, including shorter recordings.
            subject_tie = dict(zip(subjects, rng.random_sample(len(subjects))))
            while len(selected) < self.count:
                eligible = [s for s in subjects if current[s] < len(by_subject[s])]
                subject = max(eligible, key=lambda s: (
                    (epoch+1)*self.count*subject_sizes[s]/len(dataset.inputs)-subject_used[s]-current[s],
                    subject_tie[s]))
                add_subject(subject)
            subject_used.update(current)
            plan = sorted(selected)
            rng.shuffle(plan)
            used[plan] += 1
            self.plans.append(plan)
        if not self.plans or np.any(used == 0):
            raise ValueError('Epoch budget too short to expose all training groups; increase epochs')

    def set_epoch(self, epoch):
        if not 0 <= epoch < len(self.plans):
            raise ValueError('Epoch outside precomputed exposure plan')
        self.epoch = epoch

    def __iter__(self):
        return iter(self.plans[self.epoch])

    def __len__(self):
        return self.count


def exposure_report(dataset, sampler, batch_size, reference_batch_size):
    old_frames = len(dataset.inputs) * dataset.clip_length
    new_frames = len(sampler) * dataset.sequence_length
    epochs, cumulative = [], Counter()
    for epoch, plan in enumerate(sampler.plans):
        counts = Counter(i for g in plan for i in dataset.groups[g])
        cumulative.update(counts)
        epochs.append(dict(epoch=epoch, group_indices=plan,
                           unique_clips=len(counts), repeated_clip_uses=sum(counts.values())-len(counts),
                           omitted_clips=[dataset.inputs[i] for i in range(len(dataset.inputs)) if i not in counts],
                           subjects=sorted({dataset.group_subjects[g] for g in plan})))
    per_subject = {}
    for subject in sorted(set(dataset.group_subjects)):
        indices = {i for s, group in zip(dataset.group_subjects, dataset.groups) if s == subject for i in group}
        expected = len(sampler.plans) * len(indices)
        actual = sum(cumulative[i] for i in indices)
        per_subject[subject] = dict(cached_clips=len(indices), reference_clip_uses=expected,
                                   actual_clip_uses=actual, exposure_change_percent=100*(actual/expected-1))
    return dict(cached_clip_length=dataset.clip_length, training_sequence_length=dataset.sequence_length,
                cached_clips=len(dataset.inputs), candidate_groups=len(dataset), sequences_per_epoch=len(sampler),
                reference_batch_size=reference_batch_size, training_batch_size=batch_size,
                reference_frames_per_step=reference_batch_size*dataset.clip_length,
                training_frames_per_step=batch_size*dataset.sequence_length,
                reference_steps_per_epoch=math.ceil(len(dataset.inputs)/reference_batch_size),
                training_steps_per_epoch=math.ceil(len(sampler)/batch_size),
                reference_frames_per_epoch=old_frames, training_frames_per_epoch=new_frames,
                frame_exposure_change_percent=100*(new_frames/old_frames-1),
                reference_last_batch_frames=((len(dataset.inputs)-1)%reference_batch_size+1)*dataset.clip_length,
                training_last_batch_frames=((len(sampler)-1)%batch_size+1)*dataset.sequence_length,
                groups=[dict(subject=s, inputs=[dataset.inputs[i] for i in g])
                        for s, g in zip(dataset.group_subjects, dataset.groups)],
                cumulative_clip_uses={path: cumulative[i] for i, path in enumerate(dataset.inputs)},
                per_subject=per_subject,
                epochs=epochs)


def build_training_loader(dataset, config, seed_worker, generator, seed=100):
    """Called only for the training loader; the existing test loader is untouched."""
    common = dict(num_workers=config.NUM_WORKERS, batch_size=config.TRAIN.BATCH_SIZE,
                  worker_init_fn=seed_worker, generator=generator)
    if config.TRAIN.SEQUENCE_LENGTH == 0:
        # Optional training-only augmentations (both off by default):
        #   TRAIN.SPEED_AUG (SpeedAugDataset) and TRAIN.RM_AUG (RhythmMamba's data_augmentation).
        from dataset.data_loader.SpeedAugDataset import maybe_wrap
        from dataset.data_loader.RhythmMambaAugDataset import maybe_wrap as maybe_wrap_rm
        train_set, collate = maybe_wrap_rm(maybe_wrap(dataset, config), config)
        if collate is not None:
            common['collate_fn'] = collate
        return DataLoader(train_set, shuffle=True, **common)
    if config.TRAIN.DATA.DATASET != 'UBFC-rPPG' or config.TRAIN.DATA.DO_PREPROCESS:
        raise ValueError('Long-sequence experiment requires the existing UBFC cache, DO_PREPROCESS=False')
    if config.MODEL.NAME != 'BiMambaPhysKD' or config.DISTILL.WEIGHT != 0:
        raise ValueError('Controlled long-sequence experiment uses the existing no-KD student')
    cached = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
    reference_batch = config.TRAIN.SEQUENCE_REFERENCE_BATCH_SIZE
    if config.TRAIN.BATCH_SIZE * config.TRAIN.SEQUENCE_LENGTH != reference_batch * cached:
        raise ValueError('Training frames per optimizer step must equal the reference')
    joined = ConsecutiveClipDataset(dataset, cached, config.TRAIN.SEQUENCE_LENGTH)
    sampler = ExposureMatchedSampler(joined, config.TRAIN.EPOCHS, seed)
    report = exposure_report(joined, sampler, config.TRAIN.BATCH_SIZE, reference_batch)
    if report['reference_steps_per_epoch'] != report['training_steps_per_epoch']:
        raise ValueError('Cannot preserve optimizer-step count with this dataset and sequence size')
    report['csv_sha256'] = {}
    for split in ('TRAIN', 'TEST'):
        csv_path = Path(config[split].DATA.FILE_LIST_PATH)
        report['csv_sha256'][split] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    destination = Path(config.MODEL.MODEL_DIR) / 'sequence_exposure.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print('Long-sequence training: %d sequences, %d steps/epoch; frame exposure change %+.3f%%. Audit: %s' %
          (len(sampler), report['training_steps_per_epoch'], report['frame_exposure_change_percent'], destination), flush=True)
    return DataLoader(joined, sampler=sampler, **common)
