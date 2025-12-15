from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

RUN_DIRS = [
    ROOT / "runs" / "detect",
    ROOT / "runs" / "classify",
    ROOT / "runs_mvp",
]

def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except:
        return None

def safe_read_yaml(p):
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except:
        return None

def find_runs(base: Path):
    if not base.exists():
        return []
    # Ultralytics: results.csv + args.yaml dentro de cada carpeta de run
    out = []
    for d in base.rglob("*"):
        if d.is_dir() and (d / "results.csv").exists() and (d / "args.yaml").exists():
            out.append(d)
    return out

def main():
    rows = []
    for base in RUN_DIRS:
        for run in find_runs(base):
            args = safe_read_yaml(run / "args.yaml") or {}
            df = safe_read_csv(run / "results.csv")
            last = df.iloc[-1].to_dict() if df is not None and len(df) else {}

            task = args.get("task", "")
            model = args.get("model", "")
            imgsz = args.get("imgsz", "")
            epochs = args.get("epochs", "")
            batch = args.get("batch", "")
            data = args.get("data", "")

            best = run / "weights" / "best.pt"
            rows.append({
                "run_path": str(run),
                "task": task,
                "model": model,
                "imgsz": imgsz,
                "epochs": epochs,
                "batch": batch,
                "data": data,
                "best_pt": str(best) if best.exists() else "",
                # deja columnas genéricas; varían por task
                **{f"metric_{k}": v for k, v in last.items()}
            })

    out = ROOT / "runs_mvp" / "experiment_registry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
