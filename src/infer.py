import os, time, json, joblib, argparse, logging
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import load

# Use the definitive, refactored feature module
from feature_module import FeatureExtractor
from metrics import smape

# Structured logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main(args):
    """Main inference orchestrator."""
    models_dir = os.path.join(args.out_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # === 1. Load Artifacts ===
    logger.info("Loading trained artifacts...")
    fe: FeatureExtractor = load(os.path.join(models_dir, "feature_extractor.joblib"))
    meta_full = load(os.path.join(models_dir, "meta_full.joblib"))
    meta_unit = load(os.path.join(models_dir, "meta_unit.joblib"))
    kmeans = load(os.path.join(models_dir, "cluster_model.joblib"))
    
    # <<< FIX: Load the SVD transformer and the bucket classifier
    svd = load(os.path.join(models_dir, "svd_transformer.joblib"))
    bucket_clf = load(os.path.join(models_dir, "bucket_clf.joblib"))
    # <<< END FIX

    with open(os.path.join(models_dir, "cluster_stats.json"), "r") as f:
        cluster_stats = json.load(f)
    
    # Load all fold models
    fold_models = {}
    config = fe.config
    for fold in range(config["cv_folds"]):
        fold_models[fold] = {
            "lgb_full": load(os.path.join(models_dir, f"fold{fold}_lgb_full.joblib")),
            "lgb_unit": load(os.path.join(models_dir, f"fold{fold}_lgb_unit.joblib")),
            "txt_full": load(os.path.join(models_dir, f"fold{fold}_txt_full.joblib")),
            "txt_unit": load(os.path.join(models_dir, f"fold{fold}_txt_unit.joblib")),
            "img_full": load(os.path.join(models_dir, f"fold{fold}_img_full.joblib")),
            "img_unit": load(os.path.join(models_dir, f"fold{fold}_img_unit.joblib")),
        }

    # === 2. Load Test Data & Generate Features ===
    logger.info(f"Loading test data from {args.test}...")
    # NOTE: Remember to remove nrows for a full run!
    test_df = pd.read_csv(args.test, nrows=10)
    features = fe.transform(test_df)

    # === 3. Assemble Final Feature Matrix for GBDT ===
    # <<< FIX: This block is now updated to match train_stack.py
    logger.info("Applying TruncatedSVD to sparse text features...")
    classic_text_reduced = svd.transform(features["classic_text_features"])

    X_gbdt_test = np.hstack([
        classic_text_reduced,
        features["text_embeddings"],
        features["image_embeddings"],
        features["image_cues"],
        features["brand_similarities"],
        features["quantity_values"].reshape(-1, 1)
    ])
    # <<< END FIX
    logger.info(f"Assembled GBDT test feature matrix with shape: {X_gbdt_test.shape}")

    # === 4. Base Model Predictions (Averaged over folds) ===
    logger.info("Generating base model predictions...")
    base_preds = {k: np.zeros(len(test_df)) for k in ["gb_full", "gb_unit", "txt_full", "txt_unit", "img_full", "img_unit"]}
    
    for fold in range(config["cv_folds"]):
        models = fold_models[fold]
        base_preds["gb_full"] += models["lgb_full"].predict(X_gbdt_test)
        base_preds["gb_unit"] += models["lgb_unit"].predict(X_gbdt_test)
        base_preds["txt_full"] += models["txt_full"].predict(features["text_embeddings"])
        base_preds["txt_unit"] += models["txt_unit"].predict(features["text_embeddings"])
        base_preds["img_full"] += models["img_full"].predict(features["image_embeddings"])
        base_preds["img_unit"] += models["img_unit"].predict(features["image_embeddings"])

    for k in base_preds:
        base_preds[k] /= config["cv_folds"]

    # === 5. Meta-Learner Prediction ===
    logger.info("Generating meta-learner predictions...")
    # <<< FIX: Add bucket probability features
    logger.info("Generating bucket probabilities for meta-features...")
    bucket_proba = bucket_clf.predict_proba(X_gbdt_test)
    
    meta_features_full = np.hstack([
        np.vstack([base_preds["gb_full"], base_preds["txt_full"], base_preds["img_full"]]).T,
        bucket_proba
    ])
    meta_features_unit = np.hstack([
        np.vstack([base_preds["gb_unit"], base_preds["txt_unit"], base_preds["img_unit"]]).T,
        bucket_proba
    ])
    # <<< END FIX
    
    y_full_log = meta_full.predict(meta_features_full)
    y_unit_log = meta_unit.predict(meta_features_unit)

    # === 6. Final Price Composition & Clamping ===
    logger.info("Composing final price and applying clamping...")
    price_pred = np.where(
        features["quantity_values"] > 0,
        np.exp(y_unit_log) * features["quantity_values"],
        np.exp(y_full_log)
    )

    cluster_proxy_features = np.hstack([
        features["text_embeddings"][:, :128],
        features["image_embeddings"][:, :128]
    ])
    cids_test = kmeans.predict(cluster_proxy_features)
    
    for i, cid in enumerate(cids_test):
        stats = cluster_stats.get(str(cid))
        if stats:
            price_pred[i] = np.clip(price_pred[i], stats["p1"], stats["p99"])

    price_pred = np.clip(price_pred, 0.01, None)

    # === 7. Save Submission & Evaluate ===
    submission = pd.DataFrame({"sample_id": test_df["sample_id"], "price": price_pred})
    submission.to_csv(args.out_csv, index=False)
    logger.info(f"Submission file saved to {args.out_csv}")

   # In infer.py
    if args.ground_truth:
        gt_df = pd.read_csv(args.ground_truth)
        
        # <<< FIX: Isolate only the needed columns and rename explicitly
        gt_subset = gt_df[['sample_id', 'price']].copy()
        gt_subset.rename(columns={'price': 'price_true'}, inplace=True)
        
        # Merge the submission with the clean ground truth subset
        merged = pd.merge(submission, gt_subset, on='sample_id')
        
        if len(merged) == 0:
            logger.error("Ground truth merge failed. Check sample_ids.")
        else:
            # Added a debug print to show the final columns
            logger.debug(f"Merged DataFrame columns: {merged.columns.tolist()}")
            
            # Access the 'price' column from submission and 'price_true' from ground truth
            final_smape = smape(merged["price_true"], merged["price"])
            logger.info(f"SMAPE SCORE ON '{os.path.basename(args.test)}': {final_smape:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True, help="Path to test.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where artifacts are saved")
    parser.add_argument("--out_csv", type=str, default="test_out.csv", help="Path to save submission file")
    parser.add_argument("--ground_truth", type=str, help="(Optional) Path to test.csv with prices for local validation")
    main(parser.parse_args())