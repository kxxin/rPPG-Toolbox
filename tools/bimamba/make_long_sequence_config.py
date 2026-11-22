"""Derive the 640-frame no-KD experiment from the previous run's resolved config."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import yaml


def derive(reference):
    cfg = copy.deepcopy(reference)
    if (cfg['MODEL']['NAME'] != 'BiMambaPhysKD' or cfg['MODEL']['BIMAMBAPHYS']['DEPTH'] != 6
            or not cfg['MODEL']['BIMAMBAPHYS']['DUAL_PATH'] or cfg['DISTILL']['WEIGHT'] != 0):
        raise ValueError('Reference must be the 6+6 no-KD control')
    if not cfg['TEST']['USE_LAST_EPOCH'] or cfg['TRAIN']['DATA']['PREPROCESS']['CHUNK_LENGTH'] != 160:
        raise ValueError('Expected last-epoch control with 160-frame training cache')
    if cfg['TRAIN'].get('SEQUENCE_LENGTH', 0):
        raise ValueError('Reference must be the original short-sequence run')
    old_batch = cfg['TRAIN']['BATCH_SIZE']
    if old_batch % 4:
        raise ValueError('Reference batch must be divisible by four to match frame count')
    cfg['BASE'] = ['']
    cfg['TRAIN']['BATCH_SIZE'] = old_batch // 4
    cfg['TRAIN']['SEQUENCE_LENGTH'] = 640
    cfg['TRAIN']['SEQUENCE_REFERENCE_BATCH_SIZE'] = old_batch
    cfg['TRAIN']['MODEL_FILE_NAME'] += '_T640'
    cfg['LOG']['PATH'] = cfg['LOG']['PATH'].rstrip('/') + '_t640'
    # get_config appends the experiment directory; resolved configs already contain it.
    for split in ('TRAIN', 'TEST'):
        data = cfg[split]['DATA']
        cached = Path(data['CACHED_PATH'])
        if cached.name != data['EXP_DATA_NAME'] or not data['EXP_DATA_NAME']:
            raise ValueError('Expected resolved CACHED_PATH ending in EXP_DATA_NAME')
        if data['DO_PREPROCESS']:
            raise ValueError('Reference must use the existing cache, not reprocess')
        data['CACHED_PATH'] = str(cached.parent)
    cfg.pop('VALID', None)  # Unused with USE_LAST_EPOCH=True.
    cfg.pop('UNSUPERVISED', None)
    cfg['MODEL'].pop('MODEL_DIR', None)
    cfg['TEST'].pop('OUTPUT_SAVE_DIR', None)
    # Affects CPU loading only, not model, optimizer, data or evaluation.
    cfg['NUM_WORKERS'] = 4
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output config: ' + str(args.output))
    reference_bytes = args.reference_config.read_bytes()
    reference = yaml.safe_load(reference_bytes)
    cfg = derive(reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('# Generated from the previous no-KD resolved config.\n' + yaml.safe_dump(cfg, sort_keys=False))
    provenance = dict(reference_config=str(args.reference_config),
                      reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                      reference_batch=reference['TRAIN']['BATCH_SIZE'], training_batch=cfg['TRAIN']['BATCH_SIZE'],
                      previous_sequence_length=160, training_sequence_length=640,
                      note='Architecture, loss weights, epochs, LR, cache CSVs and test protocol copied from reference.')
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2))
    print('Created', args.output, '\nOutput run:', cfg['LOG']['PATH'])


if __name__ == '__main__':
    main()
