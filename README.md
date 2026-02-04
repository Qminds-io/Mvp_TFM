# MVP - Detección y Clasificación de Lesiones Mamarias (CBIS-DDSM)

Este proyecto implementa un pipeline de dos etapas para mamografías:
1) **Detección** de lesiones (masa vs calcificación) con YOLO.
2) **Clasificación** de cada ROI detectada (benigno vs maligno).

El flujo está orientado al dataset **CBIS-DDSM** y a generar un MVP reproducible para el TFM.

## Contenido del repositorio
- `dataset/`: datos y derivados.
  - `csv/`: metadatos CBIS-DDSM.
  - `jpeg/`: imágenes JPEG exportadas desde DICOM.
  - `processed_yolo/`: dataset YOLO (detección) generado.
  - `processed_cls/`: dataset clasificación generado.
- `scripts/`: scripts del pipeline (preprocesado, inferencia, evaluación y plots).
- `runs/`: entrenamientos Ultralytics (detect/classify).
- `runs_mvp/`: entrenamientos y salidas específicas del MVP.
- `yolo11*.pt`: pesos base YOLO.

## Requisitos
- Python 3.12 recomendado.
- Dependencias en `requirements.txt`.

Instalación rápida:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Preparación del dataset
Se asume que `dataset/` contiene los CSV originales de CBIS-DDSM y la carpeta `jpeg/` con imágenes exportadas desde DICOM.

Scripts útiles:
```bash
# Inspeccionar CSV
python scripts\00_inspect_csv.py

# Crear índice de imágenes (opcional, diagnóstico)
python scripts\01_build_file_index.py

# Verificar mapeo CSV -> JPEG/mascaras (diagnóstico)
python scripts\02_verify_mapping.py
```

## Generación de datasets

### Detección (YOLO)
Genera `dataset/processed_yolo` con etiquetas YOLO (2 clases: `mass`, `calcification`).
```bash
python scripts\03_make_yolo_cropped.py
```

Salida principal:
- `dataset/processed_yolo/images/{train,val,test}`
- `dataset/processed_yolo/labels/{train,val,test}`
- `dataset/processed_yolo/data.yaml`

### Clasificación (benigno/maligno)
Genera `dataset/processed_cls` con clases `benign` y `malignant`.
```bash
python scripts\05_make_cls_from_crops.py
```

## Entrenamiento (referencia)
Los entrenamientos se hicieron con Ultralytics y quedan en `runs/` y `runs_mvp/`.
Ejemplos (ajusta rutas y modelos según tu entorno):
```bash
# Detección
yolo task=detect mode=train model=yolo11s.pt data=dataset/processed_yolo/data.yaml imgsz=1024 epochs=80 batch=16

# Clasificación
yolo task=classify mode=train model=yolo11s-cls.pt data=dataset/processed_cls imgsz=224 epochs=40 batch=128
```

## Inferencia del pipeline
Usa el detector + clasificador y genera:
- JSON con resultados.
- Imagen anotada con bbox y diagnóstico por lesión.

```bash
python scripts\06_pipeline_infer.py --image mamografia.png
```

Por defecto usa:
- Detector: `runs/detect/train2/weights/best.pt`
- Clasificador: `runs_mvp/cls_yolo11s_224_e40_b128/weights/best.pt`

Los resultados salen en `runs_mvp/pipeline_outputs/`.

## Evaluación en test
Evalúa la cadena completa en `dataset/processed_yolo/images/test`:
```bash
python scripts\07_eval_pipeline_test.py --save_annotated
```

Genera:
- `runs_mvp/pipeline_eval/pipeline_test_results.csv`
- `runs_mvp/pipeline_eval/pipeline_test_summary.json`
- Anotadas en `runs_mvp/pipeline_eval/annotated` (si se habilita).

## Plots y reportes
```bash
# Plots de entrenamiento (detect/classify)
python scripts\08_plot_training_results.py

# Plots de evaluación del pipeline
python scripts\09_plot_pipeline_eval.py

# Tabla de experimentos desde runs/*
python scripts\10_make_experiment_table.py
```

## Notas importantes
- Este MVP es **experimental** y no es un dispositivo médico.
- Los umbrales del pipeline están en `scripts/06_pipeline_infer.py` y `scripts/07_eval_pipeline_test.py` (por ejemplo `CONF_MASS`, `CONF_CALC`, `MALIGN_THRESHOLD`).
- La división `train/val` es determinista por `patient_id`.

## Estructura rápida de scripts
- `00_inspect_csv.py`: inspección de CSV.
- `01_build_file_index.py`: índice de imágenes JPEG.
- `02_verify_mapping.py`: sanity check de mapeo CSV → JPEG/ROI.
- `03_make_yolo_cropped.py`: dataset YOLO detección.
- `05_make_cls_from_crops.py`: dataset clasificación.
- `06_pipeline_infer.py`: inferencia end-to-end.
- `07_eval_pipeline_test.py`: evaluación en test.
- `08_plot_training_results.py`: plots de entrenamiento.
- `09_plot_pipeline_eval.py`: plots de evaluación.
- `10_make_experiment_table.py`: registro de experimentos.
