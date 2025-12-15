from pathlib import Path
import json, time, argparse
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DET = ROOT / "runs" / "detect" / "train2" / "weights" / "best.pt"
DEFAULT_CLS = ROOT / "runs_mvp" / "cls_yolo11s_224_e40_b128" / "weights" / "best.pt"

TEST_DIR = ROOT / "dataset" / "processed_yolo" / "images" / "test"
CSV_DIR  = ROOT / "dataset" / "csv"

# Thresholds (los puedes cambiar por CLI)
CONF_MASS = 0.10
CONF_CALC = 0.25
IOU_NMS   = 0.70
MALIGN_THRESHOLD = 0.50
PAD_FRAC = 0.12
CLS_IMGSZ = 224

ID2TYPE = {0: "mass", 1: "calcification"}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_with_padding(img, x1, y1, x2, y2, pad_frac=0.12):
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)

    xx1 = clamp(int(x1) - pad_x, 0, w - 1)
    yy1 = clamp(int(y1) - pad_y, 0, h - 1)
    xx2 = clamp(int(x2) + pad_x, 1, w)
    yy2 = clamp(int(y2) + pad_y, 1, h)
    crop = img[yy1:yy2, xx1:xx2]
    return crop, (xx1, yy1, xx2, yy2)

def classify_crop(model_cls, crop_gray):
    if crop_gray.ndim == 2:
        crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = crop_gray

    res = model_cls.predict(source=crop_rgb, imgsz=CLS_IMGSZ, verbose=False)[0]
    top1 = int(res.probs.top1)
    top1conf = float(res.probs.top1conf)
    names = res.names  # {id:name}
    pred_name = str(names.get(top1, str(top1)))

    prob_malignant = None
    inv = {v: k for k, v in names.items()}
    if "malignant" in inv:
        mid = inv["malignant"]
        prob_malignant = float(res.probs.data[mid])
    else:
        prob_malignant = top1conf if pred_name.lower() == "malignant" else (1.0 - top1conf)

    diagnosis = "malignant" if prob_malignant >= MALIGN_THRESHOLD else "benign"
    return diagnosis, prob_malignant, pred_name, top1conf

def parse_filename(fname: str):
    """
    Formato que generamos:
      P_00001_mass_1_CC_LEFT_<hash>.jpg
      P_00038_calcification_1_CC_RIGHT_<hash>.jpg
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    # patient_id tiene forma P_00001 -> dos partes
    if len(parts) < 7:
        return None
    patient_id = "_".join(parts[0:2])         # P_00001
    ab_type = parts[2]                        # mass / calcification
    ab_id = parts[3]                          # "1"
    view = parts[4]                           # CC / MLO
    side = parts[5]                           # LEFT / RIGHT
    return patient_id, ab_type, ab_id, view, side

def label_from_pathology(pathology: str):
    p = str(pathology).upper().strip()
    if "MALIGNANT" in p:
        return "malignant"
    return "benign"

def build_gt_lookup():
    # Lee los 4 CSV y crea lookup por (patient_id, abnormality type, abnormality id, image view, left/right breast)
    files = [
        CSV_DIR / "mass_case_description_train_set.csv",
        CSV_DIR / "mass_case_description_test_set.csv",
        CSV_DIR / "calc_case_description_train_set.csv",
        CSV_DIR / "calc_case_description_test_set.csv",
    ]
    lut = {}
    for f in files:
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            key = (
                str(r["patient_id"]).strip(),
                str(r["abnormality type"]).strip().lower(),
                str(r["abnormality id"]).strip(),
                str(r["image view"]).strip(),
                str(r["left or right breast"]).strip().upper(),
            )
            gt = label_from_pathology(r["pathology"])
            lut[key] = gt
    return lut

def pick_detection(det_res, true_type):
    """
    Elige la mejor bbox (máxima conf) que coincida con el tipo verdadero.
    """
    boxes = det_res.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_conf = -1.0
    for b in boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        pred_type = ID2TYPE.get(cls_id, f"class_{cls_id}")
        if pred_type != true_type:
            continue
        # umbral por tipo
        if pred_type == "mass" and conf < CONF_MASS:
            continue
        if pred_type == "calcification" and conf < CONF_CALC:
            continue
        if conf > best_conf:
            best_conf = conf
            best = b
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(DEFAULT_DET))
    ap.add_argument("--cls", default=str(DEFAULT_CLS))
    ap.add_argument("--test_dir", default=str(TEST_DIR))
    ap.add_argument("--out_dir", default=str(ROOT / "runs_mvp" / "pipeline_eval"))
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--save_annotated", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotated"
    if args.save_annotated:
        ann_dir.mkdir(parents=True, exist_ok=True)

    gt_lut = build_gt_lookup()

    model_det = YOLO(args.det)
    model_cls = YOLO(args.cls)

    imgs = sorted(Path(args.test_dir).glob("*"))
    if args.max_images and args.max_images > 0:
        imgs = imgs[:args.max_images]

    rows = []
    t0 = time.time()

    for i, p in enumerate(imgs, 1):
        meta = parse_filename(p.name)
        if meta is None:
            rows.append({"image": p.name, "status": "bad_filename"})
            continue

        patient_id, ab_type, ab_id, view, side = meta
        true_type = ab_type.lower()

        gt_key = (patient_id, true_type, ab_id, view, side)
        gt_label = gt_lut.get(gt_key, None)

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append({"image": p.name, "status": "unreadable"})
            continue

        det_t = time.time()
        det_res = model_det.predict(source=str(p), imgsz=1024, conf=0.05, iou=IOU_NMS, device=0, verbose=False)[0]
        det_ms = int((time.time() - det_t) * 1000)

        best_det = pick_detection(det_res, true_type)
        if best_det is None:
            rows.append({
                "image": p.name,
                "patient_id": patient_id,
                "type": true_type,
                "gt": gt_label,
                "pred": None,
                "prob_malignant": None,
                "det_found": 0,
                "det_conf": None,
                "det_ms": det_ms,
                "cls_ms": None,
                "status": "miss"
            })
            continue

        det_conf = float(best_det.conf.item())
        x1, y1, x2, y2 = map(float, best_det.xyxy[0].tolist())

        crop, _ = crop_with_padding(img, x1, y1, x2, y2, PAD_FRAC)
        if crop.size == 0:
            rows.append({"image": p.name, "status": "empty_crop"})
            continue

        cls_t = time.time()
        pred_diag, prob_m, raw_name, top1conf = classify_crop(model_cls, crop)
        cls_ms = int((time.time() - cls_t) * 1000)

        # opcional: guardar annotated
        if args.save_annotated:
            ann = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            label = f"{true_type} | pred:{pred_diag} | det:{det_conf:.2f} m:{prob_m:.2f}"
            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(ann, label, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(ann_dir / f"{p.stem}_ann.jpg"), ann)

        rows.append({
            "image": p.name,
            "patient_id": patient_id,
            "type": true_type,
            "gt": gt_label,
            "pred": pred_diag,
            "prob_malignant": prob_m,
            "det_found": 1,
            "det_conf": det_conf,
            "det_ms": det_ms,
            "cls_ms": cls_ms,
            "status": "ok"
        })

        if i % 50 == 0:
            print(f"[{i}/{len(imgs)}] processed...")

    df = pd.DataFrame(rows)

    # Métricas
    total = len(df)
    detected = int(df["det_found"].fillna(0).sum())
    miss = int((df["status"] == "miss").sum())

    # Accuracy end-to-end (cuenta miss como incorrecto)
    eval_df = df.dropna(subset=["gt"])
    correct_e2e = int(((eval_df["gt"] == eval_df["pred"]) & (eval_df["det_found"] == 1)).sum())
    acc_e2e = correct_e2e / len(eval_df) if len(eval_df) else None

    # Accuracy condicionado a detección
    det_ok = eval_df[eval_df["det_found"] == 1]
    correct_cond = int((det_ok["gt"] == det_ok["pred"]).sum())
    acc_cond = correct_cond / len(det_ok) if len(det_ok) else None

    # Confusion matrix (solo donde hubo predicción)
    labels = ["benign", "malignant"]
    cm = {l: {m: 0 for m in labels} for l in labels}
    for _, r in det_ok.iterrows():
        if r["gt"] in labels and r["pred"] in labels:
            cm[r["gt"]][r["pred"]] += 1

    summary = {
        "n_total_images": total,
        "n_with_gt": int(len(eval_df)),
        "n_detected": detected,
        "n_missed": miss,
        "detection_rate": detected / total if total else None,
        "acc_end_to_end": acc_e2e,
        "acc_given_detected": acc_cond,
        "confusion_matrix": cm,
        "paths": {
            "detector": args.det,
            "classifier": args.cls,
            "test_dir": args.test_dir,
            "out_dir": str(out_dir)
        },
        "runtime_sec": round(time.time() - t0, 2)
    }

    out_csv = out_dir / "pipeline_test_results.csv"
    out_json = out_dir / "pipeline_test_summary.json"
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
