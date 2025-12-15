from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "dataset" / "csv"
JPEG_DIR = ROOT / "dataset" / "jpeg"
OUT = ROOT / "dataset" / "processed_yolo"

FILES = [
    ("train", CSV_DIR / "mass_case_description_train_set.csv"),
    ("test",  CSV_DIR / "mass_case_description_test_set.csv"),
    ("train", CSV_DIR / "calc_case_description_train_set.csv"),
    ("test",  CSV_DIR / "calc_case_description_test_set.csv"),
]

# 2 clases (útil para tu “contar y medir por tipo”)
CLASS_MAP = {"mass": 0, "calcification": 1}

def split_multiline_paths(s: str):
    if pd.isna(s):
        return []
    return [line.strip() for line in str(s).splitlines() if line.strip()]

def parse_uids_and_idx(dcm_rel_path: str):
    p = Path(dcm_rel_path.replace("\\", "/"))
    parts = p.parts
    if len(parts) < 4:
        return None, None, None
    uid_a = parts[-3]
    uid_b = parts[-2]
    stem = Path(parts[-1]).stem
    try:
        idx = int(stem)  # 000000->0, 000001->1
    except:
        idx = 0
    return uid_a, uid_b, idx

def build_uid_index():
    img_ext = {".jpg", ".jpeg", ".png"}
    idx = {}
    for p in JPEG_DIR.rglob("*"):
        if p.suffix.lower() in img_ext:
            uid = p.parent.name
            idx.setdefault(uid, []).append(p)
    for uid in idx:
        idx[uid] = sorted(idx[uid], key=lambda x: x.name)
    print("UID folders indexed:", len(idx))
    return idx

def resolve_jpg(dcm_candidates, uid_index):
    for dcm_rel in dcm_candidates:
        uid_a, uid_b, k = parse_uids_and_idx(dcm_rel)
        if not uid_a or not uid_b:
            continue
        for uid in (uid_b, uid_a):
            if uid in uid_index:
                files = uid_index[uid]
                if not files:
                    continue
                if k < len(files):
                    return files[k]
                return files[0]
    return None

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def bbox_to_yolo(x1,y1,x2,y2,w,h):
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh

def val_split(patient_id: str, ratio=0.2):
    # split determinístico por paciente
    h = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < ratio

def ensure_dirs():
    for s in ["train", "val", "test"]:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

def write_yaml():
    yaml = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: mass
  1: calcification
"""
    (OUT / "data.yaml").write_text(yaml, encoding="utf-8")

def main():
    ensure_dirs()
    uid_index = build_uid_index()

    total, kept, skipped = 0, 0, 0

    for split0, csv_file in FILES:
        df = pd.read_csv(csv_file)
        for _, r in df.iterrows():
            total += 1

            patient_id = str(r["patient_id"])
            ab_type = str(r["abnormality type"]).strip().lower()
            if ab_type not in CLASS_MAP:
                skipped += 1
                continue
            cls = CLASS_MAP[ab_type]

            # train -> train/val por paciente; test se queda test
            if split0 == "train":
                split = "val" if val_split(patient_id, ratio=0.2) else "train"
            else:
                split = "test"

            crop_paths = split_multiline_paths(r["cropped image file path"])
            mask_paths = split_multiline_paths(r["ROI mask file path"])

            img_p = resolve_jpg(crop_paths, uid_index)
            msk_p = resolve_jpg(mask_paths, uid_index)
            if img_p is None or msk_p is None:
                skipped += 1
                continue

            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                skipped += 1
                continue
            if img.shape != msk.shape:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            bb = bbox_from_mask(msk)
            if bb is None:
                skipped += 1
                continue

            x1,y1,x2,y2 = bb
            h, w = img.shape[:2]
            xc,yc,bw,bh = bbox_to_yolo(x1,y1,x2,y2,w,h)

            # nombre único por fila (evita colisiones)
            ab_id = str(r.get("abnormality id", "0"))
            view = str(r.get("image view", "NA"))
            side = str(r.get("left or right breast", "NA"))
            uid = hashlib.md5(f"{patient_id}_{ab_type}_{ab_id}_{view}_{side}_{img_p}".encode("utf-8")).hexdigest()[:16]
            out_name = f"{patient_id}_{ab_type}_{ab_id}_{view}_{side}_{uid}{img_p.suffix.lower()}"

            out_img = OUT / "images" / split / out_name
            out_lbl = OUT / "labels" / split / (Path(out_name).stem + ".txt")

            # copiamos la imagen tal cual (cropped ya viene en jpeg/)
            shutil.copy2(img_p, out_img)

            out_lbl.write_text(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            kept += 1

    write_yaml()
    print(f"Done. Total rows: {total}, Kept: {kept}, Skipped: {skipped}")
    print("YOLO dataset at:", OUT)
    print("YAML:", OUT / 'data.yaml')

if __name__ == "__main__":
    main()
