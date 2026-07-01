"""
preprocess_ubfc_phys_standalone.py   (streaming + selectable detector, BVP-only)
================================================================================
Standalone preprocessing for UBFC-PHYS that mirrors the rPPG-Toolbox pipeline
(DiffNormalized video + face crop + chunking), WITHOUT loading the whole dataset
and WITHOUT loading a whole full-resolution video into RAM. Focused on rPPG: it
produces the cropped, difference-normalized frames and the BVP label.

Face detector is selectable with --detector:
    haar   : OpenCV Haar cascades only (default cascade -> bundled alt2 -> alt).
    yolo   : YOLO5Face only (needs torch + the toolbox env + YOLO weights).
    auto   : Haar first, fall back to YOLO5Face only if Haar finds nothing. (default)

For every video it SCANS the first --detect_search_frames frames and uses the
first frame where the chosen detector finds a face; that box is reused for the
whole video (static crop, matching the toolbox). Falls back to the full frame
(lower quality) and says so if nothing is found.

Output is toolbox-compatible:
    <index>_input<k>.npy   shape (CHUNK_LENGTH, H, W, 3)   # DiffNormalized frames
    <index>_label<k>.npy   shape (CHUNK_LENGTH,)           # BVP, DiffNormalized

------------------------------------------------------------------------------
USAGE (run from the rPPG-Toolbox root so the default cascade path resolves):

    python preprocess_ubfc_phys_standalone.py \
        --data_path   "/mnt/c/downloads2026/cs/capstone/draft_stress/RawData" \
        --cached_path "/home/racha/PreprocessedData" \
        --detector yolo --yolo_device cuda:0 --overwrite --dry_run

Flags: --detector {haar,yolo,auto}  --overwrite  --dry_run
       --delete_raw  --tasks T1 T2 T3  --yolo_device cuda:0
------------------------------------------------------------------------------
"""
import argparse
import csv
import glob
import os
import re
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass


# ----------------------------------------------------------------------------
# Signal / label helpers
# ----------------------------------------------------------------------------
def read_signal_csv(path):
    """Read a single-column signal csv -> 1D float array.
    Robust to a few non-numeric header rows."""
    vals = []
    with open(path, "r") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                vals.append(float(row[0]))
            except ValueError:
                continue
    return np.asarray(vals)


def resample_signal(input_signal, target_length):
    return np.interp(
        np.linspace(1, input_signal.shape[0], target_length),
        np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
        input_signal,
    )


def diff_normalize_data(data):
    """Frame-to-frame normalized difference (vectorized)."""
    diff = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
    std = np.std(diff)
    diff = diff / std if std > 0 else diff * 0.0
    out = np.concatenate([diff, np.zeros((1, *data.shape[1:]), dtype=np.float32)], axis=0)
    out[np.isnan(out)] = 0
    return out.astype(np.float32)


def diff_normalize_label(label):
    diff = np.diff(label, axis=0)
    std = np.std(diff)
    diff = diff / std if std > 0 else diff * 0.0
    out = np.append(diff, np.zeros(1), axis=0)
    out[np.isnan(out)] = 0
    return out


def standardized_data(data):
    """Z-score the whole frame tensor. Use for RhythmMamba (its Fusion_Stem
    does its own internal frame differencing, so it expects standardized input)."""
    data = data - np.mean(data)
    std = np.std(data)
    data = data / std if std > 0 else data * 0.0
    data[np.isnan(data)] = 0
    return data.astype(np.float32)


def standardized_label(label):
    label = label - np.mean(label)
    std = np.std(label)
    out = label / std if std > 0 else label * 0.0
    out[np.isnan(out)] = 0
    return out


def transform_data(frames, data_type):
    if data_type == "DiffNormalized":
        return diff_normalize_data(frames)
    if data_type == "Standardized":
        return standardized_data(frames)
    return frames.astype(np.float32)  # Raw


def transform_label(bvp, label_type):
    if label_type == "DiffNormalized":
        return diff_normalize_label(bvp)
    return standardized_label(bvp)  # Standardized


def chunk_1d(sig, chunk_length, n_clips):
    return [sig[i * chunk_length:(i + 1) * chunk_length] for i in range(n_clips)]


# ----------------------------------------------------------------------------
# Detector primitives
# ----------------------------------------------------------------------------
def _largest_face(detector, rgb):
    zones = detector.detectMultiScale(rgb[:, :, :3].astype(np.uint8))
    if len(zones) < 1:
        return None
    if len(zones) >= 2:
        return list(zones[int(np.argmax(zones[:, 2]))])
    return list(zones[0])


def _yolo_box(yolo, rgb):
    res = yolo.detect_face(rgb[:, :, :3].astype(np.uint8))
    if res is None:
        return None
    x_min, y_min, x_max, y_max = res
    w, h = x_max - x_min, y_max - y_min
    cx, cy = x_min + w // 2, y_min + h // 2
    s = max(w, h)
    return [cx - s // 2, cy - s // 2, s, s]


def _enlarge(coor, coef):
    coor = list(coor)
    coor[0] = max(0, coor[0] - (coef - 1.0) / 2 * coor[2])
    coor[1] = max(0, coor[1] - (coef - 1.0) / 2 * coor[3])
    coor[2] = coef * coor[2]
    coor[3] = coef * coor[3]
    return coor


def determine_face_box(video_file, primary, search_frames, larger_box_coef, fallback=None):
    cap = cv2.VideoCapture(video_file)
    first_rgb, found, found_name, searched = None, None, None, 0
    ok, frame = cap.read()
    while ok and searched < search_frames:
        if frame is not None:
            rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
            if first_rgb is None:
                first_rgb = rgb
            for name, fn in primary:
                box = fn(rgb)
                if box is not None:
                    found, found_name = box, name
                    break
            if found is not None:
                break
            searched += 1
        ok, frame = cap.read()
    cap.release()

    if found is not None:
        return np.asarray(_enlarge(found, larger_box_coef), dtype="int"), \
               f"face via {found_name} (scanned {searched + 1} frame[s])"
    if fallback and first_rgb is not None:
        for name, fn in fallback:
            box = fn(first_rgb)
            if box is not None:
                return np.asarray(_enlarge(box, larger_box_coef), dtype="int"), f"face via {name} (fallback)"
    if first_rgb is None:
        return None, "no readable frames"
    h, w = first_rgb.shape[0], first_rgb.shape[1]
    return np.asarray([0, 0, h, w], dtype="int"), "NO FACE FOUND -> full frame (lower quality)"


def stream_crop_resize(video_file, box, width, height):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    out, corrupt = [], 0
    ok, frame = cap.read()
    while ok:
        if frame is None:
            corrupt += 1
            ok, frame = cap.read()
            continue
        rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        crop = rgb[max(box[1], 0):min(box[1] + box[3], rgb.shape[0]),
                   max(box[0], 0):min(box[0] + box[2], rgb.shape[1])]
        if crop.size == 0:
            corrupt += 1
        else:
            out.append(cv2.resize(crop, (width, height),
                                  interpolation=cv2.INTER_AREA).astype(np.float32))
        ok, frame = cap.read()
    cap.release()
    if not out:
        return np.empty((0, height, width, 3), dtype=np.float32), corrupt
    return np.asarray(out, dtype=np.float32), corrupt


# ----------------------------------------------------------------------------
# Detector construction
# ----------------------------------------------------------------------------
def build_haar_detectors(primary_cascade):
    paths = [primary_cascade]
    if getattr(cv2, "data", None) is not None:
        for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_alt.xml"):
            p = os.path.join(cv2.data.haarcascades, name)
            if os.path.exists(p) and p not in paths:
                paths.append(p)
    dets = []
    for p in paths:
        if os.path.exists(p):
            clf = cv2.CascadeClassifier(p)
            dets.append((f"haar:{os.path.basename(p)}", (lambda c: (lambda rgb: _largest_face(c, rgb)))(clf)))
    if not dets:
        sys.exit(f"No usable Haar cascade found (looked for {paths}).")
    return dets


def load_yolo(device):
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
    yolo = YOLO5Face("Y5F", device)
    return [("yolo5face", (lambda rgb: _yolo_box(yolo, rgb)))]


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Standalone, memory-light UBFC-PHYS rPPG preprocessing.")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--cached_path", required=True)
    ap.add_argument("--detector", choices=["haar", "yolo", "auto"], default="auto")
    ap.add_argument("--yolo_device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--cascade", default=os.path.join(".", "dataset", "haarcascade_frontalface_default.xml"))
    ap.add_argument("--chunk_length", type=int, default=160)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--large_box_coef", type=float, default=1.5)
    ap.add_argument("--detect_search_frames", type=int, default=60)
    ap.add_argument("--data_type", choices=["DiffNormalized", "Standardized", "Raw"], default="DiffNormalized",
                    help="Frame representation. Use 'Standardized' for RhythmMamba.")
    ap.add_argument("--label_type", choices=["DiffNormalized", "Standardized"], default="DiffNormalized",
                    help="BVP label representation. Use 'Standardized' for RhythmMamba.")
    ap.add_argument("--save_raw", action="store_true",
                    help="Also save cropped raw frames as <index>_raw<k>.npy for visualization "
                         "comparison (not used for training).")
    ap.add_argument("--store_uint8", action="store_true",
                    help="Store INPUT frames as raw uint8 (0-255), ~4x smaller than float32. "
                         "Normalize (Standardize/DiffNormalize) at TRAIN time instead. Lossless.")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--delete_raw", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    haar_dets = build_haar_detectors(args.cascade) if args.detector in ("haar", "auto") else []
    yolo_dets = None
    if args.detector in ("yolo", "auto"):
        try:
            yolo_dets = load_yolo(args.yolo_device)
            print(f"YOLO5Face loaded on {args.yolo_device}.")
        except Exception as e:
            if args.detector == "yolo":
                sys.exit(f"--detector yolo requested but YOLO5Face is unavailable: {e}\n"
                         f"Install it in the toolbox env or use --detector haar.")
            print(f"YOLO5Face unavailable ({e}); 'auto' will use Haar only.")

    if args.detector == "haar":
        primary, fallback = haar_dets, None
    elif args.detector == "yolo":
        primary, fallback = yolo_dets, None
    else:
        primary, fallback = haar_dets, yolo_dets
    print(f"Detector mode: {args.detector}  (primary={[n for n,_ in primary]}, "
          f"fallback={[n for n,_ in fallback] if fallback else None})")
    if args.store_uint8:
        print("Storage: INPUT frames as uint8 raw (~4x smaller); normalize at train time. "
              f"label_type={args.label_type}.")
    else:
        print(f"Representation: data_type={args.data_type}, label_type={args.label_type}  "
              f"(use Standardized/Standardized for RhythmMamba)")

    os.makedirs(args.cached_path, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(args.data_path, "s*", "*.avi")))
    if not vids:
        sys.exit("No videos found. --data_path must directly contain s1/, s2/, ...")

    print(f"Found {len(vids)} videos. Output -> {args.cached_path}")
    if args.delete_raw and not args.dry_run:
        print("WARNING: --delete_raw is ON. Raw .avi + bvp .csv removed after caching.\n")

    file_list, done, skipped, failed, fullframe = [], 0, 0, 0, 0

    for vid in vids:
        index = re.search(r"vid_(.*)\.avi", os.path.basename(vid)).group(1)
        if args.tasks and not any(t in index for t in args.tasks):
            continue
        existing = glob.glob(os.path.join(args.cached_path, f"{index}_input*.npy"))
        if existing and not args.overwrite:
            print(f"[skip] {index}: already cached (use --overwrite to redo)")
            skipped += 1
            continue

        bvp_file = os.path.join(os.path.dirname(vid), f"bvp_{index}.csv")
        print(f"[proc] {index}")
        try:
            box, status = determine_face_box(vid, primary, args.detect_search_frames,
                                             args.large_box_coef, fallback)
            print(f"    {status}")
            if box is None:
                raise RuntimeError("no readable frames (codec/ffmpeg) — not deleting raw.")
            if "full frame" in status:
                fullframe += 1

            frames, corrupt = stream_crop_resize(vid, box, args.size, args.size)
            if corrupt:
                print(f"    note: skipped {corrupt} corrupt frame(s)")
            if frames.shape[0] == 0:
                raise RuntimeError("0 usable frames — not deleting raw.")
            n_frames = frames.shape[0]

            bvps = transform_label(resample_signal(read_signal_csv(bvp_file), n_frames), args.label_type)
            L = args.chunk_length
            n_clips = n_frames // L
            if n_clips == 0:
                del frames
                raise RuntimeError(f"too short for chunk_length={L} — not deleting raw.")
            if args.store_uint8:
                # store raw uint8 frames (4x smaller); normalize at train time (lossless)
                f_clips = [frames[i * L:(i + 1) * L].astype(np.uint8) for i in range(n_clips)]
            else:
                data = transform_data(frames, args.data_type)
                f_clips = chunk_1d(data, L, n_clips)   # views into `data` (freed next iteration)
            # raw chunks as uint8 COPIES (not views) so the float frame buffer can be freed
            raw_clips = ([frames[i * L:(i + 1) * L].astype(np.uint8) for i in range(n_clips)]
                         if args.save_raw else None)
            del frames
            b_clips = chunk_1d(bvps, L, n_clips)

            if args.dry_run:
                extra = " (+raw)" if raw_clips is not None else ""
                print(f"    would write {n_clips} chunks: input{f_clips[0].shape}, label({args.chunk_length},){extra}")
            else:
                if existing:
                    for old in (existing
                                + glob.glob(os.path.join(args.cached_path, f"{index}_label*.npy"))
                                + glob.glob(os.path.join(args.cached_path, f"{index}_raw*.npy"))):
                        os.remove(old)
                for k in range(n_clips):
                    ip = os.path.join(args.cached_path, f"{index}_input{k}.npy")
                    lp = os.path.join(args.cached_path, f"{index}_label{k}.npy")
                    np.save(ip, f_clips[k])
                    np.save(lp, b_clips[k])
                    file_list.append(ip)
                    if raw_clips is not None:
                        np.save(os.path.join(args.cached_path, f"{index}_raw{k}.npy"), raw_clips[k])
                print(f"    wrote {n_clips} chunks of shape {f_clips[0].shape}"
                      + (" (+raw)" if raw_clips is not None else ""))

            if args.delete_raw:
                if args.dry_run:
                    print(f"    would delete raw: {os.path.basename(vid)} + bvp csv")
                else:
                    os.remove(vid)
                    if os.path.exists(bvp_file):
                        os.remove(bvp_file)
                    print("    deleted raw video + bvp csv")
            done += 1

        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

    if file_list and not args.dry_run:
        list_dir = os.path.join(args.cached_path, "DataFileLists")
        os.makedirs(list_dir, exist_ok=True)
        list_csv = os.path.join(list_dir, "standalone_filelist.csv")
        write_header = not os.path.exists(list_csv)
        with open(list_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["", "input_files"])
            start = sum(1 for _ in open(list_csv)) - 1
            for i, p in enumerate(file_list):
                w.writerow([start + i, p])
        print(f"\nFile list updated: {list_csv}")

    print(f"\nDone. processed={done}  skipped={skipped}  failed={failed}  full_frame_fallback={fullframe}")
    if fullframe:
        print("Tip: re-run those with --detector yolo --overwrite, or raise --detect_search_frames.")


if __name__ == "__main__":
    main()
