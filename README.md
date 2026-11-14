[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Mrudula1013/LesionAware-MangoNet)

[![DOI](https://zenodo.org/badge/1457560490.svg)](https://doi.org/10.5281/zenodo.24217000)

# LesionAware MangoNet

**LesionAware MangoNet: A Lightweight CNN–Transformer Framework for Robust Multi-Class Mango Leaf Disease Diagnosis with Cross-Dataset Generalization**

LesionAware MangoNet is a PyTorch-based research implementation for multi-class mango leaf disease diagnosis. It integrates a lightweight convolutional neural network (CNN), Transformer-based feature learning, multi-scale feature extraction, channel–spatial attention, weakly supervised contextual lesion attention, and geometry-aligned consistency learning.

The framework is designed to investigate disease classification, cross-dataset generalization, robustness against image corruption, model explainability, confidence calibration, and efficient deployment.

## Key Features

- Lightweight CNN–Transformer architecture for mango leaf disease classification.
- Multi-scale convolutional feature extraction using 3 × 3, 5 × 5, and 7 × 7 kernels.
- Channel and spatial attention mechanisms.
- Weakly supervised contextual lesion attention.
- Geometry-aligned masked consistency learning.
- Leakage-aware, group-based dataset splitting.
- Five-seed reproducible training configuration.
- Temperature scaling and five-view test-time augmentation (TTA).
- Four zero-shot source-to-target transfer experiments.
- Robustness evaluation under different image corruptions.
- Grad-CAM, Score-CAM, and contextual lesion visualizations.
- Ablation studies and comparative baseline evaluation.
- Structural pruning and fine-tuning.
- ONNX export and static INT8 post-training quantization (PTQ).
- Computational profiling and publication-quality figure generation.

## Datasets

The implementation supports three publicly available mango leaf disease datasets.

| Dataset | Published images | Experimental role |
|---|---:|---|
| MangoLeafBD | 4,000 | Primary source dataset |
| MLD24 | 6,400 | Secondary source dataset |
| MLD-Field | 2,336 | Independent field holdout |

**Dataset sources:**

1. [MangoLeafBD – Mendeley Data](https://data.mendeley.com/datasets/hxsnvwty3r/1)
2. [MLD24 – Mendeley Data](https://data.mendeley.com/datasets/6dvpywm2m2)
3. [MLD-Field – Mendeley Data](https://data.mendeley.com/datasets/7ghdbftp54)

The dataset images are not included in this repository. Download the original datasets from their respective sources and follow their licensing and citation requirements.

Only the published processed original field images from MLD-Field should be used. Separately published synthetic or augmented variants must not be included.

### Disease Categories

The framework supports eight classes in the following fixed order:

1. Anthracnose
2. Bacterial Canker
3. Cutting Weevil
4. Die Back
5. Gall Midge
6. Healthy
7. Powdery Mildew
8. Sooty Mould

Common label variations, including Dieback and Sooty Mold, are mapped automatically.

## Repository Structure

```text
LesionAware-MangoNet/
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── configs/
│   ├── datasets.yaml
│   ├── experiments.yaml
│   ├── model.yaml
│   ├── tiny_model.yaml
│   └── training.yaml
│
├── notebooks/
│   └── colab_quickstart.ipynb
│
├── scripts/
│   ├── prepare_data.py
│   ├── make_demo_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_transfers.py
│   ├── run_robustness.py
│   ├── explain.py
│   ├── run_ablations.py
│   ├── run_baselines.py
│   ├── compress_model.py
│   ├── evaluate_compressed.py
│   ├── predict.py
│   ├── generate_figures.py
│   └── run_all.py
│
├── src/
│   └── mangonet/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── compression/
│       ├── utils/
│       └── config.py
│
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_geometry.py
│   ├── test_metrics_data.py
│   ├── test_corruptions_explain.py
│   └── test_script_integration.py
│
├── MANUSCRIPT_IMPLEMENTATION_MAPPING.md
├── VALIDATION_REPORT.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python 3.10 or later.
- PyTorch 2.2 or later.
- Compatible torchvision installation.
- NumPy, SciPy, scikit-learn, Pandas and OpenCV.
- Sufficient storage for the original datasets.
- CUDA-compatible GPU recommended for full training.

CPU execution is supported, although full experiments may take considerably longer.

### Environment Setup

```bash
git clone https://github.com/Mrudula1013/LesionAware-MangoNet

cd LesionAware-MangoNet

python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

Install the project and dependencies:

```bash
python -m pip install --upgrade pip

pip install -e ".[all]"
```

Run the test suite:

```bash
pytest -q
```

For core training without optional export and baseline dependencies:

```bash
pip install -e .
```

## Dataset Preparation

Organize the downloaded datasets using the following structure:

```text
data/
└── raw/
    ├── MangoLeafBD/
    │   ├── Anthracnose/
    │   ├── Bacterial Canker/
    │   ├── Cutting Weevil/
    │   ├── Die Back/
    │   ├── Gall Midge/
    │   ├── Healthy/
    │   ├── Powdery Mildew/
    │   └── Sooty Mould/
    │
    ├── MLD24/
    │   └── ...
    │
    └── MLD-Field/
        └── ...
```

Nested folders beneath individual disease classes are supported.

An optional public-API dataset downloader is provided:

```bash
python scripts/prepare_data.py \
    --data-root data \
    --download MangoLeafBD MLD24 MLD-Field \
    --extract
```

If automatic downloading is unavailable, download the datasets manually and place the extracted files in the corresponding directories.

## Data Quality Control and Leakage Prevention

Run the preprocessing pipeline:

```bash
python scripts/prepare_data.py --data-root data --seed 42
```

The preprocessing pipeline performs:

- Image readability and integrity checks.
- Minimum image-size validation.
- Image sharpness assessment.
- Exact duplicate detection using SHA-256.
- Near-duplicate candidate detection using perceptual hashing.
- Rotation- and flip-aware duplicate analysis.
- Conservative duplicate grouping.
- Group-based, class-stratified train/validation/test splitting.

The primary source datasets use an approximately 70/15/15 grouped train/validation/test split. Actual proportions depend on the integrity of leaf groups.

MLD-Field is reserved exclusively for independent testing and is not used for training, validation, calibration, or quantization calibration.

Generated preprocessing outputs include:

| File | Description |
|---|---|
| `manifest.csv` | Image paths, labels, dataset identities, groups and split assignments |
| `actual_split_counts.csv` | Actual dataset and class distributions |
| `duplicate_review.csv` | Potential duplicate candidates and grouping decisions |
| `qc_summary.json` | Quality-control summary |
| `rejected.csv` | Rejected or manually excluded images |

**Important:** Near-duplicate detection requires manual review. Perceptual hashing cannot establish biological leaf identity with certainty. Review suspicious duplicate candidates and blur flags before publication. Confirmed duplicate pairs can be supplied through `--confirmed-pairs-csv`, and non-leaf images can be excluded through `--exclude-file`.

## Model Architecture

LesionAware MangoNet combines lightweight convolutional feature extraction with Transformer-based contextual representation learning.

### CNN Feature Extraction

The CNN component incorporates:

- Depthwise separable convolution.
- Multi-scale convolutional branches.
- Channel attention.
- Spatial attention.
- Residual feature connections.

### Transformer Feature Learning

The Transformer receives a class token and spatial feature tokens.

The principal configuration uses:

| Parameter | Configuration |
|---|---:|
| Input resolution | 224 × 224 |
| CNN output grid | 14 × 14 |
| Spatial tokens | 196 |
| Transformer layers | 4 |
| Embedding dimension | 256 |
| Attention heads | 4 |
| Feed-forward dimension | 512 |
| Feature fusion weight | 0.5 |

### Lesion-Aware Attention

A contextual spatial attention head generates lesion-related attention weights over the spatial feature representation.

The attention-weighted features are aggregated through multiple-instance learning (MIL) and fused with the Transformer class-token representation.

The approach provides weakly supervised contextual lesion attention without requiring pixel-level expert lesion annotations.

### Geometry-Aligned Consistency Learning

Two augmented views are generated during training using photometric and geometric transformations.

Their attention maps are aligned to the original image coordinates. Consistency is calculated over overlapping valid image regions.

The training objective combines four components:

- Cross-entropy classification loss.
- Focal loss.
- Lesion-attention entropy regularization.
- Geometry-aligned masked consistency loss.

The overall objective is:

\[
L = L_{CE} + 0.5L_{Focal}
+ 0.05rL_{Attention}
+ 0.5rL_{Consistency}
\]

where \(r\) represents the ten-epoch auxiliary-loss ramp-up schedule.

## Model Training

The main training configuration uses:

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.05 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Warmup | 5 epochs |
| Learning-rate schedule | Cosine |
| Early-stopping patience | 15 |
| Random seeds | 0, 1, 2, 3, 4 |
| Dataset split seed | 42 |

Train using the combined source datasets:

```bash
python scripts/train.py \
    --data-root data \
    --source combined \
    --seeds 0 1 2 3 4 \
    --out outputs/runs
```

Individual source-dataset experiments:

```bash
python scripts/train.py \
    --data-root data \
    --source MangoLeafBD \
    --seeds 0 1 2 3 4 \
    --out outputs/runs
```

```bash
python scripts/train.py \
    --data-root data \
    --source MLD24 \
    --seeds 0 1 2 3 4 \
    --out outputs/runs
```

The implementation records model configurations, training statistics, validation metrics, best epochs, checkpoints and experiment histories.

## Evaluation and Cross-Dataset Generalization

The framework supports in-dataset evaluation and four zero-shot source-to-target transfer experiments.

### In-Dataset Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/runs/combined/seed_0/best.pt \
    --data-root data \
    --out outputs/in_dataset/seed_0
```

### Cross-Dataset Transfer

```bash
python scripts/run_transfers.py \
    --data-root data \
    --runs outputs/runs \
    --out outputs/transfers \
    --seeds 0 1 2 3 4
```

The four transfer directions are:

1. MangoLeafBD → MLD24
2. MLD24 → MangoLeafBD
3. MangoLeafBD → MLD-Field
4. MLD24 → MLD-Field

The transfer experiments use source-trained models without target-domain retraining or recalibration.

### Confidence Calibration and Test-Time Augmentation

Temperature scaling is fitted using source-validation logits.

Five inference views are supported:

1. Original image.
2. Horizontal flip.
3. Vertical flip.
4. 90% centre crop.
5. 1.1× brightness adjustment.

Temperature calibration is applied to each view before class probabilities are averaged.

### Evaluation Metrics

The implementation supports:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Per-class precision, recall and F1-score.
- Multiclass Matthews correlation coefficient (MCC).
- Cohen's kappa.
- Eight-class confusion matrix.
- Expected calibration error (ECE).
- Brier score.
- Negative log-likelihood (NLL).

Statistical aggregation includes mean and standard deviation across seeds, t-based 95% confidence intervals, and optional paired statistical tests with Holm adjustment when comparator results are supplied.

## Robustness Evaluation

The framework evaluates model performance under multiple image corruption conditions.

Supported corruption types include:

| Corruption | Tested severity levels |
|---|---|
| Gaussian noise | 0.04–0.12 |
| Motion blur | Kernel sizes 3, 5, 7, 9, 11 |
| Brightness variation | V-channel ±0.1–0.5 |
| JPEG compression | Quality 80, 60, 40, 25, 15 |
| Random occlusion | 5%, 10%, 15%, 20%, 25% |

Run the robustness experiment:

```bash
python scripts/run_robustness.py \
    --checkpoint outputs/runs/MangoLeafBD/seed_0/best.pt \
    --data-root data \
    --target MLD-Field \
    --out outputs/robustness
```

The evaluation reports performance degradation, Macro-F1 trends and severity-wise robustness measurements.

Corrupted images are used for evaluation and are not introduced into the training loader.

## Explainability

The framework provides multiple complementary explanation techniques:

- Contextual lesion attention maps.
- Grad-CAM.
- Score-CAM.
- Internal cosine agreement.
- Deletion and insertion faithfulness evaluation.
- Average Drop and Average Increase.

Run the explanation pipeline:

```bash
python scripts/explain.py \
    --checkpoint outputs/runs/MangoLeafBD/seed_0/best.pt \
    --data-root data \
    --target MLD-Field \
    --max-images 8 \
    --out outputs/explanations
```

Faithfulness evaluation uses 20-step deletion and insertion measurements.

Lower deletion AUC and higher insertion AUC indicate better faithfulness under the implemented evaluation protocol.

**Note:** No expert pixel-level lesion masks are included. Therefore, attention and CAM agreement should not be interpreted as validated biological lesion-localization accuracy.

## Ablation Studies

The implementation supports the following architectural ablations:

| Configuration | Description |
|---|---|
| `cnn_only` | CNN-only configuration |
| `without_channel` | Removes channel attention |
| `without_spatial` | Removes spatial attention |
| `without_lesion` | Removes lesion attention |
| `without_consistency` | Removes consistency learning |
| `full` | Complete LesionAware MangoNet |

Inference ablations also examine the contribution of temperature calibration and test-time augmentation.

Run:

```bash
python scripts/run_ablations.py \
    --data-root data \
    --source combined \
    --out outputs/ablations
```

## Comparative Baseline Evaluation

The implementation supports comparison against established CNN and Transformer architectures.

Available baseline models include:

- ResNet50.
- DenseNet121.
- MobileNetV3-Large.
- EfficientNet-B0.
- Vision Transformer (ViT-B/16).
- Swin Transformer Tiny.
- MobileViT-S.
- ConvFormer-S18.
- EfficientFormer-L1.
- EdgeNeXt-Small.

Some additional architectures require the optional `timm` package and depend on the installed model availability.

Train the principal baselines:

```bash
python scripts/run_baselines.py \
    --mode train \
    --pretrained \
    --models resnet50 densenet121 mobilenet_v3_large efficientnet_b0 \
    --source combined \
    --data-root data \
    --seeds 0 1 2 3 4
```

Evaluate the trained baselines:

```bash
python scripts/run_baselines.py \
    --mode evaluate \
    --models resnet50 densenet121 mobilenet_v3_large efficientnet_b0 \
    --source combined \
    --data-root data \
    --seeds 0 1 2 3 4
```

The baseline pipeline uses consistent source-data statistics, dataset splits, training schedules and classification supervision wherever applicable.

## Model Compression and Deployment

The framework supports structural pruning, fine-tuning, ONNX export, static INT8 post-training quantization and inference profiling.

### Structural Pruning

The implementation performs physical structural pruning of CNN filters and Transformer feed-forward hidden channels.

The 256-dimensional Transformer attention embedding and attention heads are retained to preserve attention-shape compatibility.

Run:

```bash
python scripts/compress_model.py \
    --checkpoint outputs/runs/MangoLeafBD/seed_0/best.pt \
    --data-root data \
    --out outputs/compressed/seed_0
```

The pruned model is fine-tuned for 15 epochs using a learning rate of 0.0001.

The actual parameter reduction must be measured from the resulting model rather than assumed from the configured pruning percentage.

### ONNX Export and INT8 Quantization

Install the optional export dependencies:

```bash
pip install -e ".[export]"
```

The deployment pipeline supports:

- ONNX model export.
- ONNX graph validation.
- FP32 output-parity checking.
- Static QDQ INT8 post-training quantization.
- Per-channel signed weight quantization.
- Per-tensor unsigned activation quantization.
- Operator-level quantization reporting.
- CPU inference benchmarking.

Quantization calibration uses up to 512 source-training images only.

Evaluate the compressed models:

```bash
python scripts/evaluate_compressed.py \
    --original outputs/runs/MangoLeafBD/seed_0/best.pt \
    --pruned outputs/compressed/seed_0/pruned_finetuned.pt \
    --onnx-fp32 outputs/compressed/seed_0/pruned_fp32.onnx \
    --onnx-int8 outputs/compressed/seed_0/pruned_int8.onnx \
    --data-root data \
    --target MLD-Field \
    --out outputs/compression_eval
```

If ONNX dependencies are unavailable, structural pruning and PyTorch checkpoint generation can be performed without ONNX export:

```bash
python scripts/compress_model.py \
    --checkpoint outputs/runs/MangoLeafBD/seed_0/best.pt \
    --data-root data \
    --out outputs/compressed/seed_0 \
    --skip-onnx
```

Runtime acceleration depends on the hardware and execution environment. Parameter reduction alone does not establish inference speed improvement.

## Image Prediction

The framework supports single-image prediction using trained PyTorch checkpoints.

```bash
python scripts/predict.py \
    --checkpoint outputs/runs/MangoLeafBD/seed_0/best.pt \
    --image path/to/mango_leaf.jpg \
    --temperature 1.0
```

ONNX inference is also supported:

```bash
python scripts/predict.py \
    --onnx outputs/compressed/seed_0/pruned_int8.onnx \
    --stats outputs/runs/MangoLeafBD/seed_0/source_stats.json \
    --image path/to/mango_leaf.jpg
```

Prediction produces an eight-class probability distribution and the highest-confidence predicted disease category.

A temperature of 1.0 without fitted calibration represents uncalibrated inference.

The prediction output is intended for experimental research and is not a clinically or agriculturally validated treatment or pesticide recommendation.

## Publication Figure Generation

Publication-quality figures can be generated from saved experiment outputs.

```bash
python scripts/generate_figures.py \
    --history outputs/runs/combined/seed_0/history.csv \
    --metrics outputs/in_dataset/seed_0/metrics.json \
    --predictions outputs/in_dataset/seed_0/predictions.csv \
    --robustness outputs/robustness/corruptions_by_severity.csv \
    --out outputs/figures
```

The figure-generation pipeline supports:

- Training Macro-F1 curves.
- Confusion matrices.
- Reliability diagrams.
- Severity-wise robustness plots.

Figures are generated at 600 DPI from saved experimental results.

## Synthetic Demonstration

A synthetic dataset is provided through the demonstration script to verify software functionality without downloading the original datasets.

Generate synthetic images:

```bash
python scripts/make_demo_data.py \
    --data-root demo_data \
    --per-class 9
```

Prepare the demonstration data:

```bash
python scripts/prepare_data.py \
    --data-root demo_data
```

Run a small training test:

```bash
python scripts/train.py \
    --data-root demo_data \
    --source MangoLeafBD \
    --seeds 0 \
    --model-config configs/tiny_model.yaml \
    --epochs 1 \
    --batch-size 4 \
    --num-workers 0 \
    --max-train-batches 2 \
    --out outputs/demo
```

Run demonstration evaluation:

```bash
python scripts/evaluate.py \
    --checkpoint outputs/demo/MangoLeafBD/seed_0/best.pt \
    --data-root demo_data \
    --target MLD-Field \
    --max-images 8 \
    --out outputs/demo_evaluation
```


## Complete Pipeline Execution

The complete experiment orchestration is available through:

```bash
python scripts/run_all.py \
    --data-root data \
    --seeds 0 1 2 3 4
```

Optional computationally expensive experiments can be skipped using:

- `--skip-baselines`
- `--skip-explain`
- `--skip-robustness`
- `--skip-compression`

For a quick software validation run using synthetic data:

```bash
python scripts/run_all.py \
    --data-root demo_data \
    --fast-dev-run \
    --seeds 0 \
    --skip-baselines \
    --skip-onnx
```

A fast development run is a software smoke test and not a substitute for complete research experiments.

## Validation and Testing

The code package was validated in the following environment:

| Component | Environment |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| Execution environment | CPU-only |

The validation report documents **13 successfully passed pytest tests**.

The executed tests cover:

- Model forward and backward operations.
- Eight-class model output.
- Attention normalization.
- Physical pruning and gradient flow.
- Geometry-aligned consistency.
- Dataset and group-split correctness.
- Evaluation metrics and calibration.
- Image corruption generation.
- Grad-CAM and Score-CAM.


Run the test suite using:

```bash
pytest -q
```

Refer to `VALIDATION_REPORT.md` for the exact validation scope and limitations.


## Research Integrity

This implementation is intended to support reproducible research in mango leaf disease classification and cross-dataset generalization.

All scientific findings must be generated from the original datasets using the documented experimental protocols. Reported results should include the actual dataset distributions, experimental configurations, random seeds, statistical measurements and evaluation outputs.

## Acknowledgements

The datasets used in this research are publicly available through Mendeley Data. Appropriate acknowledgement and citation of the original dataset publications are required when using their images.

The implementation uses PyTorch and associated scientific Python libraries for model development, experimentation and evaluation.
