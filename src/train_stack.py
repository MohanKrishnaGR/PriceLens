import os, time, json, joblib, argparse, logging
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from joblib import dump

# --- NEW: Visualization Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------------------

from feature_module import FeatureExtractor, build_structured_targets, parse_quantity
from metrics import smape

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main(args):
    """Main training orchestrator."""
    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- NEW: Create directory for visualizations ---
    viz_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    logger.info(f"Visualizations will be saved to: {viz_dir}")
    # ---------------------------------------------

    with open(args.config, "r") as f:
        config = json.load(f)
    RANDOM_SEED = config["seed"]

    logger.info("Loading train data...")
    # NOTE: Remember to remove nrows for the full run!
    train_df = pd.read_csv(args.train) 
    
    # --- [DEBUG RECORD 0] ---
    first_record = train_df.iloc[0]
    print("\n" + "="*25 + " DEBUGGING FIRST RECORD " + "="*25)
    print(f"[DEBUG RECORD 0] Initial Data:\n{first_record}\n")
    # --- [END DEBUG] ---

    logger.info("Starting feature engineering...")
    fe = FeatureExtractor(config, img_dir=os.path.join(args.out_dir, "image_cache"))
    features = fe.fit_transform(train_df)
    dump(fe, os.path.join(models_dir, "feature_extractor.joblib"))
    logger.info("Feature engineering complete. Extractor saved.")
    
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After Feature Extraction " + "-"*20)
    print(f"[DEBUG RECORD 0] Parsed Quantity: {parse_quantity(first_record['catalog_content'])}")
    print(f"[DEBUG RECORD 0] Classic Text Features (Sparse, first 10 values): {features['classic_text_features'][0].toarray()[:, :10]}")
    print(f"[DEBUG RECORD 0] Text Embeddings (shape, first 5 values): {features['text_embeddings'][0].shape}, {features['text_embeddings'][0, :5]}")
    print(f"[DEBUG RECORD 0] Image Embeddings (shape, first 5 values): {features['image_embeddings'][0].shape}, {features['image_embeddings'][0, :5]}")
    print(f"[DEBUG RECORD 0] Image Cues: {features['image_cues'][0]}")
    print(f"[DEBUG RECORD 0] Brand Similarities (shape, all values): {features['brand_similarities'][0].shape}, {features['brand_similarities'][0]}")
    print(f"[DEBUG RECORD 0] Quantity Value: {features['quantity_values'][0]}")
    # --- [END DEBUG] ---

    targets = build_structured_targets(train_df, features)
    y_full = targets["y_full_logprice"].values
    y_unit = targets["y_unit_logprice"].values

    # --- NEW: Visualize Price Distribution ---
    logger.info("Generating price distribution plots...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df['price'], bins=50, kde=True)
    plt.title('Raw Price Distribution')
    plt.xlabel('Price')
    plt.subplot(1, 2, 2)
    sns.histplot(targets['y_full_logprice'], bins=50, kde=True, color='orange')
    plt.title('Log-Transformed Price Distribution')
    plt.xlabel('Log(Price)')
    plt.suptitle('Price Distribution Analysis')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(viz_dir, "price_distribution.png"))
    plt.close()
    # -----------------------------------------
    
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After Target Building " + "-"*20)
    print(f"[DEBUG RECORD 0] Full Log Price Target: {y_full[0]}")
    print(f"[DEBUG RECORD 0] Unit Log Price Target: {y_unit[0]}")
    # --- [END DEBUG] ---

    logger.info("Applying TruncatedSVD to sparse text features...")
    svd = TruncatedSVD(n_components=config['svd_components'], random_state=RANDOM_SEED)
    classic_text_reduced = svd.fit_transform(features["classic_text_features"])
    dump(svd, os.path.join(models_dir, "svd_transformer.joblib"))
    logger.info(f"SVD complete. Shape of reduced text features: {classic_text_reduced.shape}")
    
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After SVD Transformation " + "-"*20)
    print(f"[DEBUG RECORD 0] SVD Reduced Text Features (shape, first 5 values): {classic_text_reduced[0].shape}, {classic_text_reduced[0, :5]}")
    # --- [END DEBUG] ---

    X_gbdt = np.hstack([
        classic_text_reduced,
        features["text_embeddings"],
        features["image_embeddings"],
        features["image_cues"],
        features["brand_similarities"],
        features["quantity_values"].reshape(-1, 1)
    ])
    assert np.isfinite(X_gbdt).all(), "Non-finite values in features!"
    logger.info(f"Assembled final GBDT feature matrix with shape: {X_gbdt.shape}")
    
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After Final Matrix Assembly " + "-"*20)
    print(f"[DEBUG RECORD 0] Final GBDT Feature Vector Shape: {X_gbdt[0].shape}")
    # --- [END DEBUG] ---

    price_bins = pd.qcut(y_full, q=20, labels=False, duplicates='drop')

    logger.info("Training price-bucket classifier for meta-feature generation...")
    bucket_clf = lgb.LGBMClassifier(**config['bucket_clf'], random_state=RANDOM_SEED)
    bucket_clf.fit(X_gbdt, price_bins)
    bucket_proba = bucket_clf.predict_proba(X_gbdt)
    dump(bucket_clf, os.path.join(models_dir, "bucket_clf.joblib"))
    
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After Bucket Classifier Training " + "-"*20)
    print(f"[DEBUG RECORD 0] Price Bin Assignment: {price_bins[0]}")
    print(f"[DEBUG RECORD 0] Predicted Bucket Probabilities (shape): {bucket_proba[0].shape}\n{bucket_proba[0]}")
    # --- [END DEBUG] ---

    skf = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=RANDOM_SEED)
    oof = {k: np.zeros(len(train_df)) for k in ["gb_full", "gb_unit", "txt_full", "txt_unit", "img_full", "img_unit"]}
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_gbdt, price_bins)):
        logger.info(f"========== FOLD {fold} ==========")
        
        X_train_fold, X_val_fold, y_full_train_fold, y_full_val_fold = train_test_split(
            X_gbdt[tr_idx], y_full[tr_idx], test_size=0.2, random_state=RANDOM_SEED
        )
        _, _, y_unit_train_fold, y_unit_val_fold = train_test_split(
            X_gbdt[tr_idx], y_unit[tr_idx], test_size=0.2, random_state=RANDOM_SEED
        )

        lgb_params = config['gbdt'].copy()
        early_stopping_rounds = lgb_params.pop('early_stopping_rounds', 50)

        lgb_full = lgb.LGBMRegressor(**lgb_params, random_state=RANDOM_SEED)
        lgb_unit = lgb.LGBMRegressor(**lgb_params, random_state=RANDOM_SEED)
        
        logger.info(f"[Fold {fold}] Training GBDT (full price) with early stopping...")
        lgb_full.fit(
            X_train_fold, y_full_train_fold,
            eval_set=[(X_val_fold, y_full_val_fold)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )

        # --- NEW: Visualize Feature Importance ---
        if fold == 0:
            # Check if the model has learned anything before trying to plot
            if lgb_full.feature_importances_.sum() > 0:
                logger.info("Generating GBDT feature importance plot...")
                gbdt_feature_names = (
                    [f'svd_{i}' for i in range(classic_text_reduced.shape[1])] +
                    [f'text_emb_{i}' for i in range(features["text_embeddings"].shape[1])] +
                    [f'img_emb_{i}' for i in range(features["image_embeddings"].shape[1])] +
                    ['cue_aspect', 'cue_brightness', 'cue_color', 'cue_missing'] +
                    [f'brand_sim_{i}' for i in range(features["brand_similarities"].shape[1])] +
                    ['quantity']
                )
                lgb_full.booster_.feature_name = lambda: gbdt_feature_names
                lgb.plot_importance(lgb_full, max_num_features=30, figsize=(10, 12), title=f'Feature Importance (Fold 0)')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "gbdt_feature_importance.png"))
                plt.close()
            else:
                logger.warning(
                    "Feature importance plot not generated for Fold 0 as the model found no useful features. "
                    "This is expected on a small test dataset."
                )
        # -----------------------------------------
        
        logger.info(f"[Fold {fold}] Training GBDT (unit price) with early stopping...")
        lgb_unit.fit(
            X_train_fold, y_unit_train_fold,
            eval_set=[(X_val_fold, y_unit_val_fold)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        
        dump(lgb_full, os.path.join(models_dir, f"fold{fold}_lgb_full.joblib"))
        dump(lgb_unit, os.path.join(models_dir, f"fold{fold}_lgb_unit.joblib"))

        txt_full = MLPRegressor(**config['text_mlp'], random_state=RANDOM_SEED).fit(features['text_embeddings'][tr_idx], y_full[tr_idx])
        txt_unit = MLPRegressor(**config['text_mlp'], random_state=RANDOM_SEED).fit(features['text_embeddings'][tr_idx], y_unit[tr_idx])
        dump(txt_full, os.path.join(models_dir, f"fold{fold}_txt_full.joblib"))
        dump(txt_unit, os.path.join(models_dir, f"fold{fold}_txt_unit.joblib"))

        img_full = MLPRegressor(**config['img_mlp'], random_state=RANDOM_SEED).fit(features['image_embeddings'][tr_idx], y_full[tr_idx])
        img_unit = MLPRegressor(**config['img_mlp'], random_state=RANDOM_SEED).fit(features['image_embeddings'][tr_idx], y_unit[tr_idx])
        dump(img_full, os.path.join(models_dir, f"fold{fold}_img_full.joblib"))
        dump(img_unit, os.path.join(models_dir, f"fold{fold}_img_unit.joblib"))
        
        oof["gb_full"][va_idx] = lgb_full.predict(X_gbdt[va_idx])
        oof["gb_unit"][va_idx] = lgb_unit.predict(X_gbdt[va_idx])
        oof["txt_full"][va_idx] = txt_full.predict(features['text_embeddings'][va_idx])
        oof["txt_unit"][va_idx] = txt_unit.predict(features['text_embeddings'][va_idx])
        oof["img_full"][va_idx] = img_full.predict(features['image_embeddings'][va_idx])
        oof["img_unit"][va_idx] = img_unit.predict(features['image_embeddings'][va_idx])
        
        # --- [DEBUG RECORD 0] ---
        if 0 in va_idx:
            print("\n" + "-"*20 + f" OOF Predictions in Fold {fold} " + "-"*20)
            print(f"[DEBUG RECORD 0] GBDT Full Price Pred: {oof['gb_full'][0]}")
            # ... (rest of the debug prints)
        # --- [END DEBUG] ---
        
        fold_smape = smape(np.exp(y_full[va_idx]), np.exp(oof["gb_full"][va_idx]))
        logger.info(f"[Fold {fold}] OOF SMAPE (GB full proxy): {fold_smape:.4f}")

    # --- NEW: Visualize OOF Predictions ---
    logger.info("Generating OOF predictions plot...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_full, y=oof["gb_full"], alpha=0.5)
    min_val = min(y_full.min(), oof["gb_full"].min())
    max_val = max(y_full.max(), oof["gb_full"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title('OOF Predictions vs. True Values (GBDT Full Price)')
    plt.xlabel('True Log Price')
    plt.ylabel('Predicted Log Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, "oof_predictions_vs_true.png"))
    plt.close()
    # --------------------------------------

    logger.info("Training meta-learners...")
    meta_features_full = np.hstack([
        np.vstack([oof["gb_full"], oof["txt_full"], oof["img_full"]]).T,
        bucket_proba
    ])
    meta_features_unit = np.hstack([
        np.vstack([oof["gb_unit"], oof["txt_unit"], oof["img_unit"]]).T,
        bucket_proba
    ])
    
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After Meta-Feature Assembly " + "-"*20)
    print(f"[DEBUG RECORD 0] Meta Features Full (shape): {meta_features_full[0].shape}\n{meta_features_full[0]}")
    print(f"[DEBUG RECORD 0] Meta Features Unit (shape): {meta_features_unit[0].shape}\n{meta_features_unit[0]}")
    # --- [END DEBUG] ---
    
    meta_full = lgb.LGBMRegressor(**config['meta_learner'], random_state=RANDOM_SEED).fit(meta_features_full, y_full)
    meta_unit = lgb.LGBMRegressor(**config['meta_learner'], random_state=RANDOM_SEED).fit(meta_features_unit, y_unit)
    dump(meta_full, os.path.join(models_dir, "meta_full.joblib"))
    dump(meta_unit, os.path.join(models_dir, "meta_unit.joblib"))
    
    logger.info("Computing cluster stats for clamping...")
    cluster_proxy_features = np.hstack([
        features["text_embeddings"][:, :128],
        features["image_embeddings"][:, :128]
    ])
    kmeans = KMeans(n_clusters=config['cluster_count'], random_state=RANDOM_SEED, n_init='auto').fit(cluster_proxy_features)
    cids = kmeans.predict(cluster_proxy_features)
    price_train = np.exp(y_full)
    cluster_stats = {}
    for cid in np.unique(cids):
        vals = price_train[cids == cid]
        if len(vals) >= 10:
            cluster_stats[int(cid)] = {"p1": float(np.percentile(vals, 1)), "p99": float(np.percentile(vals, 99))}
            
    # --- [DEBUG RECORD 0] ---
    print("\n" + "-"*20 + " After Clustering " + "-"*20)
    # ... (rest of the debug prints)
    print("="*25 + "  END DEBUGGING  " + "="*25 + "\n")
    # --- [END DEBUG] ---
            
    with open(os.path.join(models_dir, "cluster_stats.json"), "w") as f:
        json.dump(cluster_stats, f)
    dump(kmeans, os.path.join(models_dir, "cluster_model.joblib"))
    logger.info(f"Clamping stats for {len(cluster_stats)} clusters saved.")
    
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save artifacts")
    main(parser.parse_args())