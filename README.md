# Multimodal Product Price Prediction System

A robust machine learning pipeline for predicting product prices using a combination of textual descriptions and product images. This system employs a **2-Level Stacking Architecture** integrating Gradient Boosted Decision Trees (LightGBM) and Deep Neural Networks (MLP), enhanced by state-of-the-art embeddings (DINOv2, SentenceTransformers).

## 🚀 Key Features

*   **Multimodal Analysis**: Combines text (TF-IDF + Semantic Embeddings) and vision (DINOv2 embeddings + Visual Cues).
*   **Deep Semantic Understanding**: Uses `sentence-transformers` for text and `facebook/dinov2` for image feature extraction.
*   **Dual-Target Strategy**: Simultaneously predicts **Full Price** and **Unit Price**, automatically selecting the best approach based on quantity parsing logic (e.g., "pack of 6").
*   **Robust Stacking**:
    *   **Level 0**: LightGBM (GBDT) + Specialized MLPs for text/image.
    *   **Level 1**: Meta-learner (LightGBM) that optimally blends base model predictions.
*   **Smart Post-Processing**: Uses K-Means clustering to "clamp" predictions within realistic price ranges for similar products.
*   **High-Performance Utilities**: Includes an async, parallel image downloader (`ultra_fast_downloader.py`) that bypasses the need for external tools like `aria2c`.

## 📂 Project Structure

```
.
├── config.json                 # Hyperparameters for models and feature engineering
├── requirements.txt            # Python dependencies
├── ultra_fast_downloader.py    # Async image downloader script
├── image_change.py             # Utility to rename images to match sample_ids
├── README.md                   # Project documentation
└── src/
    ├── feature_module.py       # Core feature extraction (Text, Image, Regex logic)
    ├── train_stack.py          # Main training pipeline (Stacking, Cross-validation)
    ├── infer.py                # Inference script for generating predictions
    └── metrics.py              # SMAPE and other metric definitions
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable).
2.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A GPU (CUDA) is highly recommended for the deep learning feature extraction steps (Torch/Transformers).*

## ⚡ Usage Workflow

### 1. Data Preparation & Image Downloading
Before training, you need to download the product images referenced in your dataset.

```bash
# Download images in parallel
python ultra_fast_downloader.py --train dataset/train.csv --test dataset/test.csv --out_dir dataset/image_cache

# Rename images to match sample_ids (Crucial for the pipeline)
python image_change.py
```
*Modify `image_change.py` variables if your CSV paths or image directory differ.*

### 2. Training the Model
The training script performs feature extraction, trains the Level 0 base models, trains the Level 1 meta-learner, and saves all artifacts.

```bash
python src/train_stack.py \
    --train dataset/train.csv \
    --config config.json \
    --out_dir artifacts
```
*   **Outputs**: Models, scalers, and cluster stats are saved to `artifacts/models`.
*   **Visualizations**: Feature importance and price distribution plots are saved to `artifacts/visualizations`.

### 3. Inference
Generate predictions on a new test set.

```bash
python src/infer.py \
    --test dataset/test.csv \
    --out_dir artifacts \
    --out_csv submission.csv
```

**Optional Validation:**
If your test file includes ground truth prices, add the `--ground_truth` flag to see the SMAPE score immediately:
```bash
python src/infer.py ... --ground_truth dataset/test.csv
```

## ⚙️ Configuration (`config.json`)

The system behavior is controlled by `config.json`. Key sections include:

*   **`cv_folds`**: Number of cross-validation folds (default: 2).
*   **`svd_components`**: Dimensions for SVD reduction of sparse text features.
*   **`gbdt`**: LightGBM hyperparameters (learning rate, depth, leaves).
*   **`text_mlp` / `img_mlp`**: Architecture for the neural network base models.
*   **`cluster_count`**: Number of clusters for the post-processing price clamping.

## 🧠 Architecture Details

1.  **Feature Extraction**:
    *   **Text**: Normalized text -> TF-IDF (Sparse) + MiniLM (Dense).
    *   **Image**: DINOv2 (Dense) + Color/Brightness Stats (Numeric).
    *   **Logic**: Regex extracts quantities (e.g., "50ml", "2 kg") to compute unit prices.

2.  **Stacking**:
    *   Base models predict `log(price)` and `log(unit_price)`.
    *   Meta-learner takes these predictions + a "Price Bucket" probability to output the final log-price.

3.  **Inference Logic**:
    *   If a quantity is detected: `Final Price = Predicted Unit Price * Quantity`.
    *   Otherwise: `Final Price = Predicted Full Price`.
    *   **Clamping**: The price is restricted to the 1st-99th percentile of prices found in its semantic cluster.

## 📋 Requirements
*   `numpy`, `pandas`, `scipy`
*   `scikit-learn`, `lightgbm`
*   `torch`, `transformers`, `sentence-transformers`
*   `Pillow`, `tqdm`
