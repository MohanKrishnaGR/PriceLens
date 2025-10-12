#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitive Multimodal Feature Extractor for Smart Product Pricing
License: MIT
Strategy:
- RETAINS all robust classic features from the original (TF-IDF, hashing, quantity parsing, brand heuristics).
- AUGMENTS text features with deep semantic embeddings (SentenceTransformer/Gemma).
- REPLACES the weak image hashing with a powerful vision model (DINOv2).
- USES a robust dictionary output to prevent brittle slicing in downstream model scripts.
"""
import os
import re
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from PIL import Image, ImageOps
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- (Normalization, Parsing, and other helpers remain the same as the original) ---
def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("pcs.", "pcs").replace("pc.", "pc")
    s = re.sub(r"\b(pack)\s*[-:]*\s*of\s*", r"\1 of ", s)
    s = re.sub(r"(\d+)\s*pcs\b", r"\1 pcs", s)
    s = re.sub(r"(\d+)\s*pc\b", r"\1 pc", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

KEYWORD_FLAGS = {"refurbished": [" refu", "renewed"], "combo": [" combo ", "bundle", " kit ", " set "], "premium": [" premium ", " deluxe ", " pro "], "organic": [" organic ", " bio "]}
MODEL_MARKERS = {"model", "series", "mk", "v", "gen"}
UNIT_MAP = {"kg": ("g", 1000.0), "g": ("g", 1.0), "mg": ("g", 0.001), "l": ("ml", 1000.0), "ml": ("ml", 1.0), "oz": ("g", 28.3495)}
UNIT_PATTERNS = {"kg": r"(\d+(?:\.\d+)?)\s*kg", "g": r"(\d+(?:\.\d+)?)\s*g", "mg": r"(\d+(?:\.\d+)?)\s*mg", "l": r"(\d+(?:\.\d+)?)\s*l", "ml": r"(\d+(?:\.\d+)?)\s*ml", "oz": r"(\d+(?:\.\d+)?)\s*oz", "count_packof": r"pack\s*of\s*(\d+)", "count_units": r"\b(\d+)\s*(?:pcs|pieces|count)\b"}

@dataclass
class QuantityResult:
    unit_label: str; qty_value: float
def parse_quantity(text: str) -> QuantityResult:
    s = normalize_text(text)
    for unit, pat in UNIT_PATTERNS.items():
        if "count" in unit: continue
        m = re.search(pat, s)
        if m:
            val = float(m.group(1)); base_unit, scale = UNIT_MAP[unit]; total = val * scale
            m_cnt = re.search(UNIT_PATTERNS["count_packof"], s) or re.search(UNIT_PATTERNS["count_units"], s)
            if m_cnt: total *= float(m_cnt.group(1))
            return QuantityResult(base_unit, total)
    m_cnt = re.search(UNIT_PATTERNS["count_packof"], s) or re.search(UNIT_PATTERNS["count_units"], s)
    if m_cnt: return QuantityResult("count", float(m_cnt.group(1)))
    return QuantityResult("", 0.0)

def candidate_brand_tokens(s: str) -> List[str]:
    toks = re.split(r"\s+", normalize_text(s)); cand = []
    if toks: cand.append(toks[0])
    for i in range(len(toks) - 1):
        if toks[i + 1] in MODEL_MARKERS: cand.append(toks[i])
    return [c for c in cand if c and not c.isdigit() and len(c) >= 2]
# --- (End of unchanged helper functions) ---

class FeatureExtractor:
    def __init__(self, config: Dict, img_dir: str = "image_cache"):
        self.config = config
        self.img_dir = img_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize all fit-dependent attributes to None
        self.tfidf_word, self.tfidf_char, self.text_scaler, self.brand_enc, self.brand_vocab = None, None, None, None, None
        self.cue_scaler, self.brand_prototypes = None, None
        self.text_embedder, self.image_embedder, self.image_processor = None, None, None
        self._initialize_embedders()
        self.text_embed_dim = self.text_embedder.get_sentence_embedding_dimension()
        self.image_embed_dim = self.image_embedder.config.hidden_size

    def _initialize_embedders(self):
        """Initializes deep learning models. Using S-BERT as a proxy for Gemma embedding models."""
        if self.text_embedder is None:
            print(f"Loading text embedder on device: {self.device}")
            self.text_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        if self.image_embedder is None:
            print(f"Loading image embedder on device: {self.device}")
            model_name = 'facebook/dinov2-base'
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.image_embedder = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _plan_image_path(self, sid: str, url: str) -> str:
        #     h = hashlib.md5((str(sid) + "|" + str(url)).encode("utf-8")).hexdigest()
        #     os.makedirs(self.img_dir, exist_ok=True)
        #     return os.path.join(self.img_dir, f"{h}.jpg")
        if sid is None or str(sid).strip() == "":
            return None

        sample_id = str(sid).strip()
        os.makedirs(self.img_dir, exist_ok=True)

        # 1️⃣ Preferred: sample_id-based naming convention (e.g. 33127.jpg)
        path = os.path.join(self.img_dir, f"{sample_id}.jpg")
        if os.path.exists(path):
            return path

        # 2️⃣ Try alternate extensions
        for ext in [".jpeg", ".png", ".webp"]:
            alt = os.path.join(self.img_dir, f"{sample_id}{ext}")
            if os.path.exists(alt):
                return alt

        # 3️⃣ Fallback: legacy MD5 hashed filename (for backward compatibility)
        key = f"{sample_id}|{url}"
        md5 = hashlib.md5(key.encode("utf-8")).hexdigest()
        legacy_path = os.path.join(self.img_dir, f"{md5}.jpg")
        return legacy_path


    def _load_image_or_none(self, path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(path): return None
            return ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        except Exception:
            return None

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        print("Fitting FeatureExtractor...")
        # --- Text and Brand Space ---
        train_text = df["catalog_content"].fillna("").astype(str).apply(normalize_text)
        self.tfidf_word = TfidfVectorizer(min_df=3, max_features=50000, ngram_range=(1, 2)).fit(train_text)
        self.tfidf_char = TfidfVectorizer(analyzer="char", min_df=3, max_features=30000, ngram_range=(3, 5)).fit(train_text)
        
        counts = pd.Series([t for s in train_text for t in candidate_brand_tokens(s)]).value_counts()
        self.brand_vocab = counts.head(250).index.tolist()
        def pick_brand(s):
            for c in candidate_brand_tokens(s):
                if c in self.brand_vocab: return c
            return "OTHER"
        train_brands = train_text.apply(pick_brand)
        self.brand_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(train_brands.values.reshape(-1, 1))
        
        # --- Scalers and Prototypes ---
        # Note: we call transform internally to get the matrix needed for fitting scalers and prototypes
        feature_blocks = self._transform_logic(df)
        self.cue_scaler = StandardScaler().fit(feature_blocks['image_cues'])
        
        brand_to_idx = {b: [] for b in self.brand_vocab}
        for idx, brand in enumerate(train_brands):
            if brand != "OTHER" and feature_blocks['image_embeddings'][idx].sum() != 0:
                brand_to_idx[brand].append(idx)
        
        protos = [feature_blocks['image_embeddings'][inds[:30]].mean(axis=0) for b, inds in brand_to_idx.items() if inds]
        self.brand_prototypes = np.stack(protos if protos else [np.zeros(self.image_embed_dim)])
        
        print("Fit complete.")
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return self._transform_logic(df)
        
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return self.fit(df).transform(df)

    def _transform_logic(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        N = len(df)
        texts = df["catalog_content"].fillna("").astype(str).apply(normalize_text)
        
        # --- [KEPT] Classic Sparse Features ---
        if self.tfidf_word: # Check if fitted
            X_w = self.tfidf_word.transform(texts)
            X_c = self.tfidf_char.transform(texts)
            def pick_brand(s):
                for c in candidate_brand_tokens(s):
                    if c in self.brand_vocab: return c
                return "OTHER"
            X_b = self.brand_enc.transform(texts.apply(pick_brand).values.reshape(-1, 1))
            sparse_features = sparse.hstack([X_w, X_c, X_b], format="csr")
        else: # Handle case where transform is called before fit (for internal fit step)
            sparse_features = sparse.csr_matrix((N, 0)) # Empty matrix
        
        # --- [NEW] Deep Semantic & Visual Embeddings ---
        text_embs = self.text_embedder.encode(texts.tolist(), batch_size=128, show_progress_bar=True, convert_to_numpy=True)
        img_embs = np.zeros((N, self.image_embed_dim), dtype=np.float32)
        img_cues = np.zeros((N, 4), dtype=np.float32)
        
        valid_images, valid_indices = [], []
        for i, (sid, url) in enumerate(zip(df["sample_id"], df["image_link"])):
            im = self._load_image_or_none(self._plan_image_path(sid, url))
            if im:
                w, h = im.size; arr = np.asarray(im, dtype=np.float32) / 255.0
                img_cues[i] = [w / max(h, 1), arr.mean(), np.abs(arr[:,:,0]-arr[:,:,1]).mean() + np.abs(0.5*(arr[:,:,0]+arr[:,:,1])-arr[:,:,2]).mean(), 0.0]
                valid_images.append(im); valid_indices.append(i)
            else:
                img_cues[i] = [1.0, 0.0, 0.0, 1.0]

        if valid_images:
            with torch.no_grad():
                for i in range(0, len(valid_images), 64):
                    batch_imgs, batch_indices = valid_images[i:i+64], valid_indices[i:i+64]
                    inputs = self.image_processor(images=batch_imgs, return_tensors="pt").to(self.device)
                    outputs = self.image_embedder(**inputs)
                    img_embs[batch_indices] = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # --- [KEPT] Final Feature Assembly ---
        scaled_cues = self.cue_scaler.transform(img_cues) if self.cue_scaler else img_cues
        qty_values = np.array([parse_quantity(s).qty_value for s in texts], dtype=np.float32)
        
        # --- [FIXED] Conditionally calculate brand similarities ---
        if self.brand_prototypes is not None:
            norm_img = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-8)
            norm_proto = self.brand_prototypes / (np.linalg.norm(self.brand_prototypes, axis=1, keepdims=True) + 1e-8)
            brand_sims = norm_img @ norm_proto.T
        else:
            # Placeholder for the fit() stage before prototypes are created
            brand_sims = np.zeros((N, 1), dtype=np.float32)

        # --- [MODIFIED] Return a clean dictionary for robust downstream use ---
        return {
            "classic_text_features": sparse_features,
            "text_embeddings": text_embs.astype(np.float32),
            "image_embeddings": img_embs.astype(np.float32),
            "image_cues": scaled_cues.astype(np.float32),
            "brand_similarities": brand_sims.astype(np.float32),
            "quantity_values": qty_values,
        }

def build_structured_targets(df: pd.DataFrame, features: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Helper to build target variables for the two-head model."""
    price = df["price"].astype(float).values
    qty = np.clip(features["quantity_values"], 1.0, None)
    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "has_quantity": (features["quantity_values"] > 0).astype(int),
        "y_full_logprice": np.log(np.maximum(price, 1e-9)),
        "y_unit_logprice": np.log(np.maximum(price / qty, 1e-9)),
    })

