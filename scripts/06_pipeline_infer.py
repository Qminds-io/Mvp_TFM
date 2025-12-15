from pathlib import Path
import json, uuid, time
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DET_MODEL = ROOT / "runs" / "detect" / "train2" / "weights" / "best.pt"
CLS_MODEL = ROOT / "runs_mvp" / "cls_yolo11s_224_e40_b128" / "weights" / "best.pt"

# Umbrales recomendados (para subir recall en mass sin dañar calc)
CONF_MASS = 0.10
CONF_CALC = 0.25
IOU_NMS = 0.70

# Umbral para decidir maligno/benigno por ROI
MALIGN_THRESHOLD = 0.50

# Padding alrededor del bbox (reduce shift respecto a "cropped images" del dataset)
PAD_FRAC = 0.12

# Tamaño esperado por el clasificador
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
    # crop_gray: np.uint8 (H,W) o (H,W,1)
    if crop_gray.ndim == 2:
        crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = crop_gray

    # Ultralytics acepta ndarray
    res = model_cls.predict(source=crop_rgb, imgsz=CLS_IMGSZ, device=0, verbose=False)[0]

    # res.probs: Probabilities object
    top1 = int(res.probs.top1)
    top1conf = float(res.probs.top1conf)

    # Mapear id->nombre (benign/malignant) según el modelo
    names = res.names  # dict {id: name}
    pred_name = str(names.get(top1, str(top1)))

    # También sacamos prob de malignant si existe explícito
    prob_malignant = None
    inv = {v: k for k, v in names.items()}
    if "malignant" in inv:
        mid = inv["malignant"]
        prob_malignant = float(res.probs.data[mid])

    return pred_name, top1conf, prob_malignant

def run_pipeline(image_path: Path, save_annotated=True, out_dir=None):
    out_dir = Path(out_dir) if out_dir else (ROOT / "runs_mvp" / "pipeline_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    request_id = str(uuid.uuid4())
    t0 = time.time()

    # Lee imagen (grayscale)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"No pude leer la imagen: {image_path}")
    h, w = img.shape[:2]

    model_det = YOLO(str(DET_MODEL))
    model_cls = YOLO(str(CLS_MODEL))

    # Detector
    det_t0 = time.time()
    det_res = model_det.predict(
        source=str(image_path),
        imgsz=1024,      # puedes bajar a 640 si quieres más velocidad
        conf=0.05,       # lo dejamos bajo y filtramos por clase luego
        iou=IOU_NMS,
        device=0,
        verbose=False
    )[0]
    det_ms = int((time.time() - det_t0) * 1000)

    boxes = det_res.boxes
    lesions = []

    # Para overlay
    ann = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls.item())
            det_conf = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())

            lesion_type = ID2TYPE.get(cls_id, f"class_{cls_id}")

            # filtro por tipo con umbrales distintos
            if lesion_type == "mass" and det_conf < CONF_MASS:
                continue
            if lesion_type == "calcification" and det_conf < CONF_CALC:
                continue

            # Medidas en px
            bw = x2 - x1
            bh = y2 - y1
            area_px = bw * bh
            diameter_px = max(bw, bh)

            # Crop + padding
            crop, (cx1, cy1, cx2, cy2) = crop_with_padding(img, x1, y1, x2, y2, PAD_FRAC)
            if crop.size == 0:
                continue

            # Clasificación
            cls_t0 = time.time()
            pred_name, pred_conf, prob_malignant = classify_crop(model_cls, crop)
            cls_ms = int((time.time() - cls_t0) * 1000)

            # Normaliza salida
            # Si el modelo devuelve "benign" o "malignant"
            if prob_malignant is None:
                # fallback: si top1 dice malignant, prob_malignant ~ pred_conf, si no, 1-pred_conf
                prob_malignant = pred_conf if pred_name.lower() == "malignant" else (1.0 - pred_conf)

            diagnosis = "malignant" if prob_malignant >= MALIGN_THRESHOLD else "benign"

            lesions.append({
                "type": lesion_type,
                "det_conf": det_conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_padded_xyxy": [cx1, cy1, cx2, cy2],
                "w_px": bw,
                "h_px": bh,
                "area_px": area_px,
                "diameter_px": diameter_px,
                "cls_pred": pred_name,
                "cls_top1_conf": pred_conf,
                "prob_malignant": prob_malignant,
                "diagnosis": diagnosis,
                "latency_ms": {"det": det_ms, "cls": cls_ms}
            })

            # Annotate
            label = f"{lesion_type} | {diagnosis} | d={det_conf:.2f} m={prob_malignant:.2f}"
            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(ann, label, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Agregación (decisión final por imagen/estudio)
    count_total = len(lesions)
    any_malign = any(l["diagnosis"] == "malignant" for l in lesions)
    final_label = "malignant" if any_malign else "benign"

    # (opcional) score final = max prob malignant
    final_score = max([l["prob_malignant"] for l in lesions], default=0.0)

    payload = {
        "request_id": request_id,
        "image": image_path.name,
        "image_path": str(image_path),
        "model_paths": {"detector": str(DET_MODEL), "classifier": str(CLS_MODEL)},
        "counts": {"total": count_total},
        "final": {"label": final_label, "score": final_score},
        "lesions": lesions,
        "timing_ms": {"total": int((time.time() - t0) * 1000)}
    }

    out_json = out_dir / f"{image_path.stem}_{request_id}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_img = None
    if save_annotated:
        out_img = out_dir / f"{image_path.stem}_{request_id}_annotated.jpg"
        cv2.imwrite(str(out_img), ann)

    print("Wrote JSON:", out_json)
    if out_img:
        print("Wrote Annotated:", out_img)
    print("Final:", final_label, "score=", f"{final_score:.3f}", "lesions=", count_total)

    return payload

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta a una imagen .jpg/.png")
    ap.add_argument("--out", default=None, help="Carpeta de salida")
    args = ap.parse_args()

    run_pipeline(Path(args.image), save_annotated=True, out_dir=args.out)
