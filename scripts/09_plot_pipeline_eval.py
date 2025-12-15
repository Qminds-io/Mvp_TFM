from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs_mvp" / "pipeline_eval" / "pipeline_test_summary.json"

def main():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))

    out = Path(s["paths"]["out_dir"]) / "plots_tfm"
    out.mkdir(exist_ok=True)

    # 1) Barra de métricas clave
    detection_rate = s["detection_rate"]
    acc_e2e = s["acc_end_to_end"]
    acc_cond = s["acc_given_detected"]

    plt.figure()
    plt.bar(["detection_rate", "acc_end_to_end", "acc_given_detected"],
            [detection_rate, acc_e2e, acc_cond])
    plt.ylim(0, 1)
    plt.title("Pipeline metrics (test)")
    plt.tight_layout()
    plt.savefig(out / "pipeline_metrics.png", dpi=200)
    plt.close()

    # 2) Confusion matrix (solo detectados)
    cm = s["confusion_matrix"]
    labels = ["benign", "malignant"]
    mat = np.array([[cm["benign"]["benign"], cm["benign"]["malignant"]],
                    [cm["malignant"]["benign"], cm["malignant"]["malignant"]]], dtype=int)

    plt.figure()
    plt.imshow(mat)
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)
    plt.title("Confusion matrix (given detected)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(mat[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix_detected.png", dpi=200)
    plt.close()

    print("Saved:", out)

if __name__ == "__main__":
    main()
