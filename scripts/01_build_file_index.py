from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JPEG_DIR = ROOT / "dataset" / "jpeg"

img_ext = {".jpg", ".jpeg", ".png"}
all_files = [p for p in JPEG_DIR.rglob("*") if p.suffix.lower() in img_ext]

print("Total images found:", len(all_files))

# Guarda un índice por nombre de archivo (basename)
out = ROOT / "dataset" / "_index_images.txt"
out.write_text("\n".join(str(p) for p in all_files), encoding="utf-8")
print("Wrote:", out)
