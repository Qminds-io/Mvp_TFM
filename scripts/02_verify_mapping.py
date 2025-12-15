from pathlib import Path
import pandas as pd
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "dataset" / "csv"
JPEG_DIR = ROOT / "dataset" / "jpeg"

MASS_TRAIN = CSV_DIR / "mass_case_description_train_set.csv"
CALC_TRAIN = CSV_DIR / "calc_case_description_train_set.csv"

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
    stem = Path(parts[-1]).stem  # "000000"
    try:
        idx = int(stem)  # 000000->0, 000001->1
    except:
        idx = 0
    return uid_a, uid_b, idx

def build_uid_index():
    # Mapea nombre de carpeta (UID) -> lista de jpg/png dentro (ordenada)
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

        # prueba primero uid_b (suele ser el folder correcto), si no, uid_a
        for uid in (uid_b, uid_a):
            if uid in uid_index:
                files = uid_index[uid]
                if len(files) == 0:
                    continue
                if k < len(files):
                    return files[k]
                else:
                    return files[0]
    return None

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def check(df_cases: pd.DataFrame, uid_index, n=80):
    ok_img = ok_msk = ok_bb = 0
    fail = 0

    for _, r in df_cases.head(n).iterrows():
        crop_paths = split_multiline_paths(r["cropped image file path"])
        mask_paths = split_multiline_paths(r["ROI mask file path"])

        img_p = resolve_jpg(crop_paths, uid_index)
        msk_p = resolve_jpg(mask_paths, uid_index)

        if img_p is None:
            fail += 1
            continue
        ok_img += 1

        if msk_p is None:
            fail += 1
            continue
        ok_msk += 1

        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            fail += 1
            continue

        if img.shape != msk.shape:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        bb = bbox_from_mask(msk)
        if bb is None:
            fail += 1
            continue
        ok_bb += 1

    print(f"Checked {n} -> imgOK:{ok_img}, maskOK:{ok_msk}, bboxOK:{ok_bb}, FAIL:{fail}")

def main():
    uid_index = build_uid_index()
    df_mass = pd.read_csv(MASS_TRAIN)
    df_calc = pd.read_csv(CALC_TRAIN)

    print("MASS:")
    check(df_mass, uid_index, n=80)
    print("CALC:")
    check(df_calc, uid_index, n=80)

if __name__ == "__main__":
    main()
