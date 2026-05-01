#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STREAMLIT DEMO: LANDSCAPE IMAGE CLUSTERING
==========================================

Interactive web app to cluster images using trained CLIP models
Displays uploaded images with cluster predictions

Usage:
  streamlit run src/processing/streamlit_demo.py
"""

import sys
sys.path.insert(0, '.')

import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import io

import streamlit as st
import torch
import clip

try:
    from src.storage.mongodb_client import MongoDBClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class ClusteringModel:
    """Wrapper for clustering model inference."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.artifact_dir = "checkpoints"
        self.cluster_labels = {}
        
        self._load_models()
        self._load_cluster_labels()
        
        self.device = "cpu"  # Use CPU for Streamlit compatibility
        self._load_clip()
    
    def _load_models(self):
        """Load KMeans, PCA, and Scaler."""
        km_path = os.path.join(self.artifact_dir, f"kmeans_{self.model_id}.pkl")
        pca_path = os.path.join(self.artifact_dir, f"pca_scaler_{self.model_id}.pkl")
        
        if not os.path.exists(km_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(f"Model files not found for {self.model_id}")
        
        with open(km_path, "rb") as f:
            self.kmeans = pickle.load(f)
        
        with open(pca_path, "rb") as f:
            pca_data = pickle.load(f)
            self.pca = pca_data["pca"]
            self.scaler = pca_data["scaler"]
    
    def _load_clip(self):
        """Load CLIP model."""
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
    
    def _load_cluster_labels(self):
        """Load cluster dominant labels from MongoDB."""
        if not MONGODB_AVAILABLE:
            return
        
        try:
            mongo = MongoDBClient()
            col_prof = mongo.db["cluster_profiles"]
            
            # Get latest profiles
            profiles = list(col_prof.find(
                {},
                {"cluster_id": 1, "dominant_label": 1, "_id": 0}
            ).sort("_id", -1).limit(self.kmeans.n_clusters))
            
            profiles = sorted(profiles, key=lambda p: p["cluster_id"])
            
            for p in profiles:
                self.cluster_labels[int(p["cluster_id"])] = p["dominant_label"]
        
        except Exception as e:
            print(f"Warning: Could not load cluster labels: {e}")
    
    def predict(self, image: Image.Image) -> dict:
        """
        Predict cluster for an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict with prediction results
        """
        # Extract CLIP features
        tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.clip_model.encode_image(tensor)
            feat = feat.cpu().numpy()[0]
        
        # L2 normalize
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        
        # Standardize and PCA
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        feat_pca = self.pca.transform(feat_scaled)
        
        # Predict cluster
        cluster_id = int(self.kmeans.predict(feat_pca)[0])
        distances = self.kmeans.transform(feat_pca)[0]
        distance = float(np.min(distances))
        
        # Confidence
        confidence = 1.0 / (1.0 + distance)
        
        # Label
        label = self.cluster_labels.get(cluster_id, "unknown")
        
        return {
            "cluster_id": cluster_id,
            "label": label,
            "distance": round(distance, 4),
            "confidence": round(confidence, 4),
        }


# ================================================================
# STREAMLIT APP
# ================================================================

def main():
    st.set_page_config(
        page_title="Landscape Image Clustering",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏔️ Landscape Image Clustering Demo")
    st.markdown("""
    Cluster landscape images using **CLIP Vision-Language Model**
    
    **Categories:** Mountain, Forest, Sea, Snow, Desert
    """)
    
    # Sidebar: Model selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_id = st.text_input(
            "Model ID (first 8 chars of run_id)",
            value="06912ab1",
            help="Example: 06912ab1"
        )
        
        st.markdown("---")
        st.markdown("""
        ### About
        - **Model:** CLIP ViT-B/32 (512-dim embeddings)
        - **Algorithm:** KMeans clustering + PCA
        - **Categories:** 5 landscape types
        - **Purity:** 97.34%
        """)
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = ClusteringModel(model_id)
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return
    
    # Main tabs
    tab1, tab2 = st.tabs(["📸 Single Image", "📁 Batch Upload"])
    
    # ================================================================
    # TAB 1: Single Image
    # ================================================================
    with tab1:
        st.header("Upload a Single Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp", "gif"],
                key="single_image"
            )
        
        with col2:
            st.subheader("Prediction Result")
            
            if uploaded_file is not None:
                # Load image
                if isinstance(uploaded_file, Image.Image):
                    image = uploaded_file
                else:
                    image = Image.open(uploaded_file).convert("RGB")
                
                # Predict
                with st.spinner("Predicting..."):
                    result = model.predict(image)
                
                # Display results
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        label="Cluster ID",
                        value=result["cluster_id"],
                    )
                
                with metric_col2:
                    st.metric(
                        label="Category",
                        value=result["label"].upper(),
                    )
                
                with metric_col3:
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']*100:.1f}%",
                    )
                
                with metric_col4:
                    st.metric(
                        label="Distance",
                        value=f"{result['distance']:.3f}",
                    )
                
                # Display image
                st.markdown("---")
                st.markdown("### Image Preview")
                
                img_col1, img_col2 = st.columns([1, 1])
                with img_col1:
                    st.image(image, caption="Uploaded Image")
                
                with img_col2:
                    # Color coding by cluster
                    color_map = {
                        0: "#8B4513",  # mountain - brown
                        1: "#228B22",  # forest - green
                        2: "#4169E1",  # sea - blue
                        3: "#F0F8FF",  # snow - alice blue
                        4: "#DAA520",  # desert - goldenrod
                    }
                    
                    color = color_map.get(result["cluster_id"], "#CCCCCC")
                    
                    st.markdown(f"""
                    <div style="
                        background-color: {color};
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        height: 100%;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <h2 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                            {result['label'].upper()}
                        </h2>
                        <p style="margin: 10px 0 0 0; color: white; font-size: 18px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                            Confidence: {result['confidence']*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ================================================================
    # TAB 2: Batch Upload
    # ================================================================
    with tab2:
        st.header("Upload Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png", "bmp", "gif"],
            accept_multiple_files=True,
            key="batch_images"
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} images selected**")
            
            # Predict button
            if st.button("🔍 Predict All", use_container_width=True, key="predict_all"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Load image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    # Predict
                    result = model.predict(image)
                    result["filename"] = uploaded_file.name
                    results.append(result)
                    
                    # Progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {idx + 1}/{len(uploaded_files)}")
                
                # Display results
                st.markdown("---")
                st.subheader("Results Summary")
                
                # Statistics
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                avg_confidence = np.mean([r["confidence"] for r in results])
                
                with stat_col1:
                    st.metric("Total Images", len(results))
                
                with stat_col2:
                    st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
                
                with stat_col3:
                    st.metric("Unique Categories", len(set(r["label"] for r in results)))
                
                # Cluster distribution
                st.markdown("### Cluster Distribution")
                
                from collections import Counter
                label_counts = Counter(r["label"] for r in results)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Bar chart
                    st.bar_chart(
                        {label: count for label, count in label_counts.items()},
                    )
                
                with col2:
                    # Pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=list(label_counts.values()),
                        names=list(label_counts.keys()),
                        title="Category Distribution",
                        hole=0.3,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.markdown("### Detailed Results")
                
                results_df = []
                for r in results:
                    results_df.append({
                        "Filename": r["filename"],
                        "Category": r["label"].capitalize(),
                        "Cluster": r["cluster_id"],
                        "Confidence": f"{r['confidence']*100:.1f}%",
                        "Distance": f"{r['distance']:.3f}",
                    })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Gallery view
                st.markdown("### Image Gallery")
                
                # Group by cluster
                grouped = {}
                for uploaded_file, result in zip(uploaded_files, results):
                    label = result["label"]
                    if label not in grouped:
                        grouped[label] = []
                    grouped[label].append((uploaded_file, result))
                
                for label in sorted(grouped.keys()):
                    with st.expander(f"📂 {label.upper()} ({len(grouped[label])} images)"):
                        cols = st.columns(4)
                        
                        for idx, (uploaded_file, result) in enumerate(grouped[label]):
                            image = Image.open(uploaded_file).convert("RGB")
                            col = cols[idx % 4]
                            
                            with col:
                                st.image(
                                    image,
                                    caption=f"{uploaded_file.name}\n{result['confidence']*100:.0f}% confidence"
                                )


if __name__ == "__main__":
    main()
