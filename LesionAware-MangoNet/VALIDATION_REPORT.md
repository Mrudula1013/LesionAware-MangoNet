# Validation Report — LesionAware MangoNet Code Package

**Build:** 2026-11-13. **Environment:** Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu. CPU-only session; no original mango dataset downloaded to this build environment.

## Executed successfully

- Python compilation (`python -m compileall -q src scripts`).
- **13 Pytest tests passed**, including 8-class model forward/backward, attention normalisation, physical pruning and gradient flow, geometry-aligned consistency and flip alignment, source/group split correctness, metrics/calibration, all 25 corruption combinations, Score-CAM/Grad-CAM, and fully automated synthetic fixture→QC manifest→trained checkpoint→field-evaluation command-line integration.
- Synthetic demonstration image generation and QC manifests for MangoLeafBD, MLD24 and MLD-Field names.
- Tiny CNN–Transformer training on synthetic images: produced `best.pt`, best epoch/validation metrics, per-source source-training stats and training history.
- Independent field-evaluation command: fitted source-validation temperature, generated predictions.csv and metrics.json for synthetic field holdout.
- Explainability command: generated real lesion, Grad-CAM and Score-CAM maps on a synthetic image, and a 600-dpi figure.
- Physical pruning command: rebuilt and transferred a tiny model with **53,229 → 46,161** parameters (**13.28% reduction** in that tiny model); passed forward/backward, ran a fine-tuning epoch and saved `pruned_finetuned.pt`. This is *not* a scientific compression result.
- MAC profiling code executed on the same tiny model via its labelled PyTorch-profiler fallback.
- Gaussian noise 5-severity robustness CLI on a synthetic test subset; metrics and severity CSV created.
- Post-compression evaluation of original and pruned PyTorch checkpoints on synthetic field images.
- Single-image inference CLI on a synthetic image, producing eight-class probability JSON.
- Ablation training (`without_lesion`) and evaluation on a tiny synthetic subset.
- torchvision `resnet50` baseline training and evaluation on a tiny synthetic subset (random initialisation for offline smoke test).
- 600-dpi figure generator from actual saved synthetic testing metrics, predictions and corruption CSV.
- End-to-end `run_all.py --fast-dev-run` orchestration: three synthetic source configurations trained, and all four source–target transfers executed successfully for seed 0.

## Not executable in this build environment

- ONNX export and ONNX Runtime static INT8 PTQ: optional `onnx` and `onnxruntime` modules were not installed and the Python package index could not be reached. The code implements these functions with explicit errors when dependencies are missing, but these stages require testing on the destination runtime.
- timm baselines: optional `timm` package absent. torchvision baselines were exercised.
- Full five-seed 100-epoch experimental results and published dataset loading: data are not bundled and have not been obtained here. The numerical examples from the synthetic smoke tests have **no validity as research findings**.
- Desktop CUDA latency and deployment on hardware such as Jetson/Raspberry Pi: no such device was tested here. Device-specific tests should be run by the user.

## Usage

After installing dependencies, run `pytest -q`. Then follow the README for actual Mendeley dataset ingestion, manual QC review, full experiments and exports. Only report scientific performance after real-data experiments complete.
