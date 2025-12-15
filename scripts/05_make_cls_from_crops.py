from pathlib import Path
import pandas as pd
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "dataset" / "csv"
JPEG_DIR = ROOT / "dataset" / "jpeg"
OUT = ROOT / "dataset" / "processed_cls"

FILES = [
    ("train", CSV_DIR / "mass_case_description_train_set.csv"),
    ("test",  CSV_DIR / "mass_case_description_test_set.csv"),
    ("train", CSV_DIR / "calc_case_description_train_set.csv"),
    ("test",  CSV_DIR / "calc_case_description_test_set.csv"),
]

def split_multiline_paths(s: str):
    if pd.isna(s):
        return []
    return [line.strip() for line in str(s).splitlines() if line.strip()]

def parse_uids_and_idx(dcm_rel_path: str):
    # .../<UID_A>/<UID_B>/000000.dcm
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

def label_from_pathology(pathology: str):
    p = str(pathology).upper().strip()
    if "MALIGNANT" in p:
        return "malignant"
    # BENIGN y BENIGN_WITHOUT_CALLBACK -> benign
    return "benign"

def val_split(patient_id: str, ratio=0.2):
    h = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < ratio

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["benign", "malignant"]:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    uid_index = build_uid_index()

    total, kept, skipped = 0, 0, 0
    for split0, csv_file in FILES:
        df = pd.read_csv(csv_file)
        for _, r in df.iterrows():
            total += 1
            patient_id = str(r["patient_id"])
            y = label_from_pathology(r["pathology"])

            # train -> train/val por paciente
            if split0 == "train":
                split = "val" if val_split(patient_id, ratio=0.2) else "train"
            else:
                split = "test"

            crop_paths = split_multiline_paths(r["cropped image file path"])
            img_p = resolve_jpg(crop_paths, uid_index)
            if img_p is None or not img_p.exists():
                skipped += 1
                continue

            # nombre único
            ab_type = str(r.get("abnormality type", "na")).strip().lower()
            ab_id = str(r.get("abnormality id", "0"))
            view = str(r.get("image view", "NA"))
            side = str(r.get("left or right breast", "NA"))
            uid = hashlib.md5(f"{patient_id}_{ab_type}_{ab_id}_{view}_{side}_{img_p}".encode("utf-8")).hexdigest()[:16]
            out_name = f"{patient_id}_{ab_type}_{ab_id}_{view}_{side}_{uid}{img_p.suffix.lower()}"

            out_img = OUT / split / y / out_name
            shutil.copy2(img_p, out_img)
            kept += 1

    print(f"Done. Total rows: {total}, Kept: {kept}, Skipped: {skipped}")
    print("Classification dataset at:", OUT)

if __name__ == "__main__":
    main()
