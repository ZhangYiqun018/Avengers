#!/usr/bin/env python3
"""
Rank Router Generation Script

This script generates the necessary artifacts for the rank-based router:
1. Model performance rankings for each cluster
2. Cluster centers for embedding-based routing
3. Embedding normalizer for consistent preprocessing

The rank router works by:
- Clustering training queries using their embeddings
- Learning model performance rankings within each cluster
- Using cluster proximity and model rankings to route queries at inference time
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import joblib
from loguru import logger
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from collections import Counter

from core.ablation.embedding_cache import EmbeddingCache


# Default model mapping (M01-M22 format used internally)
DEFAULT_MODEL_MAPPING = {
    "fin-r1-index": "M01",
    "glm-4-9b-index": "M02", 
    "eurus-2-7b-prime-index": "M03",
    "qwen2.5-7b-instruct-index": "M04",
    "qwen2.5-coder-7b-instruct-index": "M05",
    "qwen2.5-math-7b-instruct-index": "M06",
    "nemotron-8b-index": "M07",
    "medreason_8b_index": "M08",
    "gemma-2-9b-it": "M09",
    "Llama-3.1-8B-Instruct": "M10",
    "cogito-v1-preview-llama-8B": "M11",
    "internlm3-8b-instruct": "M12",
    "mistral-7b-instruct-v0.3": "M13",
    "Falcon3-7B-Instruct": "M14",
    "Phi-4-mini-instruct": "M15",
    "Granite-3.1-8B-Instruct": "M16",
    "Yi-1.5-9B-Chat": "M17",
    "OLMo-2-1124-7B-Instruct": "M18",
    "Llama-3.1-Tulu-3.1-8B": "M19",
    "Hermes-3-Llama-3.1-8B": "M20",
    "Llama-3.1-8B-UltraMedical": "M21",
    "DeepSeek-R1-Distill-Qwen-7B": "M22",
}

DEFAULT_EMBED_CONFIG = {
    "gte-qwen2-7b-instruct": {
        "url": "input your api url",
        "api_key": "input your api key",
        "model_name": "gte-qwen2-7b-instruct"
    },
    "jina-embeddings-v3": {
        "url": "input your api url",
        "api_key": "input your api key",
        "model_name": "jina-embeddings-v3"
    },
    "text-embedding-3-small": {
        "url": "https://api.openai.com/v1",
        "api_key": "your-openai-api-key",
        "model_name": "text-embedding-3-small"
    }
}


class RankRouterGenerator:
    """Generates rank router artifacts from training data."""
    
    def __init__(
        self,
        embed_config: Dict[str, str],
        model_mapping: Dict[str, str] = None,
        cache_dir: str = ".cache"
    ):
        """
        Initialize the rank router generator.
        
        Args:
            embed_config: Embedding model configuration with url, model_name, and optional api_key
            model_mapping: Dictionary mapping model names to model IDs (e.g., M01, M02, ...)  
            cache_dir: Directory for caching embeddings and intermediate results
        """
        self.embed_config = embed_config
        self.model_mapping = model_mapping or DEFAULT_MODEL_MAPPING
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding cache
        self.embedder = EmbeddingCache(
            base_url=embed_config["url"],
            api_key=embed_config.get("api_key", "sk-1234567890"),
            model_name=embed_config["model_name"],
            cache_dir=cache_dir
        )
        
        logger.info(f"Initialized RankRouterGenerator with embedding model: {embed_config['model_name']}")
    
    def load_training_data(
        self, 
        data_path: str, 
        test_size: float = 0.3,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and split training data.
        
        Args:
            data_path: Path to the training data JSON file
            test_size: Fraction of data to use for testing
            seed: Random seed for train/test split
            
        Returns:
            Tuple of (train_set, test_set) as lists of dictionaries
        """
        logger.info(f"Loading training data from: {data_path}")
        
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset_names = list(set(dataset['dataset']))
        
        # Split each dataset separately to maintain balance
        subsets = []
        for name in dataset_names:
            subset = dataset.filter(lambda x: x['dataset'] == name, num_proc=4)
            split = subset.train_test_split(test_size=test_size, seed=seed)
            subsets.append(split)
        
        # Combine splits
        train_set = concatenate_datasets([split['train'] for split in subsets])
        test_set = concatenate_datasets([split['test'] for split in subsets])
        
        train_list = train_set.to_list()
        test_list = test_set.to_list()
        
        logger.info(f"Loaded {len(train_list)} training samples and {len(test_list)} test samples")
        return train_list, test_list
    
    def generate_embeddings(
        self, 
        queries: List[str],
        save_normalizer: bool = True,
        normalizer_name: str = None
    ) -> Tuple[np.ndarray, Normalizer]:
        """
        Generate normalized embeddings for queries.
        
        Args:
            queries: List of query strings
            save_normalizer: Whether to save the normalizer to disk
            normalizer_name: Custom name for normalizer file
            
        Returns:
            Tuple of (normalized_embeddings, normalizer)
        """
        logger.info(f"Generating embeddings for {len(queries)} queries...")
        
        # Get embeddings with batching for efficiency
        embeddings = self.embedder.batch(queries, max_batch_size=100)
        embeddings = np.array(embeddings)
        
        # Normalize embeddings to unit length
        normalizer = Normalizer(norm="l2")
        normalized_embeddings = normalizer.fit_transform(embeddings)
        
        if save_normalizer:
            if normalizer_name is None:
                normalizer_name = f"ranking_embedding_normalizer_{self.embed_config['model_name'].replace('-', '_')}"
            normalizer_path = self.cache_dir / f"{normalizer_name}.joblib"
            joblib.dump(normalizer, normalizer_path)
            logger.info(f"Saved normalizer to: {normalizer_path}")
        
        return normalized_embeddings, normalizer
    
    def perform_clustering(
        self, 
        embeddings: np.ndarray, 
        n_clusters: int = 64,
        seed: int = 42,
        save_centers: bool = True,
        centers_name: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform K-means clustering on embeddings.
        
        Args:
            embeddings: Normalized embedding matrix
            n_clusters: Number of clusters
            seed: Random seed for clustering
            save_centers: Whether to save cluster centers to disk
            centers_name: Custom name for centers file
            
        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init="k-means++",
            n_init="auto",
            max_iter=1000,
            algorithm="elkan"
        )
        
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        logger.info(f"Clustering completed. Inertia: {inertia:.4f}")
        
        if save_centers:
            if centers_name is None:
                centers_name = f"ranking_centers_split_k{n_clusters}_m{len(self.model_mapping)}"
            centers_path = self.cache_dir / f"{centers_name}.npy"
            np.save(centers_path, centers)
            logger.info(f"Saved cluster centers to: {centers_path}")
        
        return labels, centers
    
    def compute_cluster_rankings(
        self, 
        labels: np.ndarray,
        train_data: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Compute model performance rankings within each cluster.
        
        Args:
            labels: Cluster labels for each training sample
            train_data: Training data with 'records' field containing model performance
            
        Returns:
            Dictionary mapping cluster_id to cluster info including rankings
        """
        logger.info("Computing model rankings within clusters...")
        
        cluster_stats = {}
        
        # Collect statistics for each cluster
        for cluster_id, sample in zip(labels, train_data):
            cluster_id = int(cluster_id)
            
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {
                    'total': 0,
                    'scores': {}
                }
            
            cluster_stats[cluster_id]['total'] += 1
            
            # Update model statistics
            records = sample['records']
            for model_name, is_correct in records.items():
                if model_name not in cluster_stats[cluster_id]['scores']:
                    cluster_stats[cluster_id]['scores'][model_name] = {
                        'correct': 0,
                        'total': 0,
                        'accuracy': 0
                    }
                
                cluster_stats[cluster_id]['scores'][model_name]['total'] += 1
                if is_correct:
                    cluster_stats[cluster_id]['scores'][model_name]['correct'] += 1
        
        # Compute rankings for each cluster
        rankings = {}
        for cluster_id, stats in cluster_stats.items():
            # Calculate accuracy for each model
            for model_name in stats['scores']:
                model_stats = stats['scores'][model_name]
                model_stats['accuracy'] = (
                    model_stats['correct'] / model_stats['total'] 
                    if model_stats['total'] > 0 else 0
                )
            
            # Sort models by accuracy (descending)
            sorted_models = sorted(
                stats['scores'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            rankings[cluster_id] = {
                'total': stats['total'],
                'scores': stats['scores'],
                'ranking': [model_name for model_name, _ in sorted_models]
            }
        
        logger.info(f"Computed rankings for {len(rankings)} clusters")
        return rankings
    
    def select_top_models(
        self, 
        rankings: Dict[int, Dict],
        n_models: int = 10,
        method: str = "frequency"
    ) -> List[str]:
        """
        Select top performing models based on cluster rankings.
        
        Args:
            rankings: Cluster rankings from compute_cluster_rankings
            n_models: Number of models to select
            method: Selection method ("frequency", "average_rank", or "mixed")
            
        Returns:
            List of selected model IDs
        """
        if method == "frequency":
            # Select models that appear most frequently in top positions
            top1_counter = Counter()
            top2_counter = Counter()
            
            for cluster_info in rankings.values():
                ranking = cluster_info['ranking']
                if len(ranking) > 0:
                    top1_counter[ranking[0]] += 1
                    top2_counter[ranking[0]] += 1
                if len(ranking) > 1:
                    top2_counter[ranking[1]] += 0.5
            
            # Select based on top-1 frequency first, then top-2
            selected = set()
            for model in top1_counter.most_common():
                if len(selected) >= n_models:
                    break
                selected.add(model[0])
            
            if len(selected) < n_models:
                for model in top2_counter.most_common():
                    if len(selected) >= n_models:
                        break
                    if model[0] not in selected:
                        selected.add(model[0])
            
            # Fill remaining slots with any available models
            all_models = list(self.model_mapping.values())
            for model in all_models:
                if len(selected) >= n_models:
                    break
                if model not in selected:
                    selected.add(model)
            
            return list(selected)[:n_models]
        
        else:
            raise NotImplementedError(f"Selection method '{method}' not implemented")
    
    def generate_artifacts(
        self,
        data_path: str,
        output_dir: str = None,
        n_clusters: int = 64,
        n_selected_models: int = 10,
        test_size: float = 0.3,
        seed: int = 42,
        save_mapping: bool = True
    ) -> Dict[str, str]:
        """
        Generate all rank router artifacts.
        
        Args:
            data_path: Path to training data JSON file
            output_dir: Output directory for artifacts (defaults to cache_dir)
            n_clusters: Number of clusters for K-means
            n_selected_models: Number of top models to select
            test_size: Test split ratio
            seed: Random seed
            save_mapping: Whether to save model mapping file
            
        Returns:
            Dictionary with paths to generated artifacts
        """
        if output_dir is None:
            output_dir = self.cache_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=== Starting Rank Router Generation ===")
        
        # 1. Load training data
        train_data, test_data = self.load_training_data(data_path, test_size, seed)
        
        # 2. Generate embeddings
        queries = [sample["query"] for sample in train_data]
        embeddings, normalizer = self.generate_embeddings(
            queries, 
            save_normalizer=True,
            normalizer_name=f"ranking_embedding_normalizer_{self.embed_config['model_name'].replace('-', '_')}"
        )
        
        # 3. Perform clustering
        labels, centers = self.perform_clustering(
            embeddings,
            n_clusters=n_clusters,
            seed=seed,
            save_centers=True,
            centers_name=f"ranking_centers_split_k{n_clusters}_m{len(self.model_mapping)}"
        )
        
        # 4. Compute cluster rankings
        rankings = self.compute_cluster_rankings(labels, train_data)
        
        # 5. Select top models
        selected_models = self.select_top_models(rankings, n_selected_models)
        logger.info(f"Selected models: {selected_models}")
        
        # 6. Save artifacts
        artifacts = {}
        
        # Save rankings
        rankings_path = output_dir / f"ranking_split_k{n_clusters}_m{len(self.model_mapping)}.json"
        with open(rankings_path, 'w', encoding='utf-8') as f:
            # Convert numpy int keys to regular int for JSON serialization
            json_rankings = {int(k): v for k, v in rankings.items()}
            json.dump(json_rankings, f, indent=2, ensure_ascii=False)
        artifacts['rankings'] = str(rankings_path)
        
        # Save model mapping
        if save_mapping:
            mapping_path = output_dir / f"map_m{len(self.model_mapping)}.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_mapping, f, indent=2, ensure_ascii=False)
            artifacts['mapping'] = str(mapping_path)
        
        # Save selected models list
        selected_path = output_dir / f"selected_models_m{n_selected_models}.json"
        with open(selected_path, 'w', encoding='utf-8') as f:
            json.dump(selected_models, f, indent=2, ensure_ascii=False)
        artifacts['selected_models'] = str(selected_path)
        
        # Add paths to other artifacts
        artifacts['centers'] = str(output_dir / f"ranking_centers_split_k{n_clusters}_m{len(self.model_mapping)}.npy")
        artifacts['normalizer'] = str(output_dir / f"ranking_embedding_normalizer_{self.embed_config['model_name'].replace('-', '_')}.joblib")
        
        logger.info("=== Rank Router Generation Complete ===")
        logger.info("Generated artifacts:")
        for name, path in artifacts.items():
            logger.info(f"  {name}: {path}")
        
        return artifacts


def main():
    """Command line interface for rank router generation."""
    parser = argparse.ArgumentParser(
        description="Generate rank router artifacts for the Avengers framework"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to training data JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="core/rank",
        help="Output directory for generated artifacts"
    )
    
    parser.add_argument(
        "--embed_model",
        type=str,
        default="gte-qwen2-7b-instruct",
        choices=list(DEFAULT_EMBED_CONFIG.keys()),
        help="Embedding model to use"
    )
    
    parser.add_argument(
        "--embed_url",
        type=str,
        help="Custom embedding API URL (overrides default)"
    )
    
    parser.add_argument(
        "--embed_api_key",
        type=str,
        help="API key for embedding model (if required)"
    )
    
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=64,
        help="Number of clusters for K-means"
    )
    
    parser.add_argument(
        "--n_models",
        type=int,
        default=10,
        help="Number of top models to select"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Fraction of data to use for testing"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache",
        help="Directory for caching embeddings"
    )
    
    args = parser.parse_args()
    
    # Setup embedding configuration
    embed_config = DEFAULT_EMBED_CONFIG[args.embed_model].copy()
    if args.embed_url:
        embed_config["url"] = args.embed_url
    if args.embed_api_key:
        embed_config["api_key"] = args.embed_api_key
    
    # Initialize generator
    generator = RankRouterGenerator(
        embed_config=embed_config,
        model_mapping=DEFAULT_MODEL_MAPPING,
        cache_dir=args.cache_dir
    )
    
    # Generate artifacts
    artifacts = generator.generate_artifacts(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        n_selected_models=args.n_models,
        test_size=args.test_size,
        seed=args.seed
    )
    
    print("\n=== Generation Summary ===")
    print(f"Successfully generated rank router artifacts in: {args.output_dir}")
    print("Use these files in your rank router configuration:")
    print(f"  centres_path: {artifacts['centers']}")
    print(f"  rankings_path: {artifacts['rankings']}")
    print(f"  mapping_path: {artifacts['mapping']}")
    print(f"  normalizer_path: {artifacts['normalizer']}")


if __name__ == "__main__":
    main()