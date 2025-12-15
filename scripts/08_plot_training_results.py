from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def plot_results(run_dir: Path, title: str):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print("No results.csv in", run_dir)
        return

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)

    out = run_dir / "plots_tfm"
    out.mkdir(exist_ok=True)

    # 1) Losses (si existen)
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    if loss_cols:
        plt.figure()
        for c in loss_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{title} - losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "losses.png", dpi=200)
        plt.close()

    # 2) Métricas típicas detect: mAP50, mAP50-95, precision, recall
    metric_candidates = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    metric_cols = [c for c in metric_candidates if c in df.columns]
    if metric_cols:
        plt.figure()
        for c in metric_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title(f"{title} - metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "metrics.png", dpi=200)
        plt.close()

    # 3) Métricas típicas cls: top1/top5 (si aparecen)
    cls_cols = [c for c in df.columns if "top1" in c.lower() or "top5" in c.lower()]
    if cls_cols:
        plt.figure()
        for c in cls_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{title} - top1/top5")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "topk.png", dpi=200)
        plt.close()

    print("Saved plots to:", out)

if __name__ == "__main__":
    # Cambia estas rutas a tus corridas
    det_run = ROOT / "runs" / "detect" / "train2"
    cls_run = ROOT / "runs_mvp" / "cls_yolo11s_224_e40_b128"

    plot_results(det_run, "Detector YOLO11s 1024 e80")
    plot_results(cls_run, "Classifier YOLO11s-cls 224 e40")
