#!/usr/bin/env python3
"""
Ablation Experiments for Avengers Rank Router

This module provides a unified framework for running ablation studies on various
components of the rank router system. It includes experiments for:

1. Number of clusters (K) sensitivity analysis
2. Clustering method comparison (K-means, Hierarchical, GMM, etc.)
3. Model selection strategies
4. Top-K cluster selection impact
5. Train/test split ratio analysis

Each experiment can be run independently with configurable parameters.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# ML and clustering imports
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from datasets import Dataset, load_dataset, concatenate_datasets

# Import shared utilities
from core.ablation.embedding_cache import EmbeddingCache
from core.ablation.utils import EMBED_CONFIG


class AblationExperimentRunner:
    """Unified runner for all ablation experiments."""
    
    def __init__(
        self,
        embed_config: Dict[str, str],
        data_path: str,
        available_models: List[str],
        dataset_names: List[str],
        cache_dir: str = ".cache",
        seed: int = 42
    ):
        """
        Initialize the ablation experiment runner.
        
        Args:
            embed_config: Embedding model configuration
            data_path: Path to training data JSON file
            available_models: List of model IDs to use (e.g., ['M01', 'M02', ...])
            dataset_names: List of dataset names to evaluate on
            cache_dir: Directory for caching embeddings
            seed: Random seed for reproducibility
        """
        self.embed_config = embed_config
        self.data_path = data_path
        self.available_models = available_models
        self.dataset_names = dataset_names
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        
        # Constants for router simulation
        self.BETA = 6.0
        self.DEFAULT_RANK = 999
        
        # Initialize embedding cache
        self.embedder = EmbeddingCache(
            base_url=embed_config["url"],
            api_key=embed_config.get("api_key", "sk-1234567890"),
            model_name=embed_config["model_name"],
            cache_dir=cache_dir
        )
        
        # Cache for embeddings (computed once, reused across experiments)
        self._cached_embeddings = None
        self._cached_normalizer = None
        self._cached_train_data = None
        self._cached_test_data = None
        
        logger.info(f"Initialized AblationExperimentRunner with {len(available_models)} models")
    
    def _load_data_if_needed(self, test_size: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """Load and cache training data."""
        if self._cached_train_data is None or self._cached_test_data is None:
            logger.info(f"Loading data from {self.data_path}")
            
            dataset = load_dataset("json", data_files=self.data_path, split="train")
            dataset_names = list(set(dataset['dataset']))
            
            subsets = []
            for name in dataset_names:
                subset = dataset.filter(lambda x: x['dataset'] == name, num_proc=4)
                split = subset.train_test_split(test_size=test_size, seed=self.seed)
                subsets.append(split)
            
            train_set = concatenate_datasets([split['train'] for split in subsets])
            test_set = concatenate_datasets([split['test'] for split in subsets])
            
            self._cached_train_data = train_set.to_list()
            self._cached_test_data = test_set.to_list()
            
            logger.info(f"Loaded {len(self._cached_train_data)} train, {len(self._cached_test_data)} test samples")
        
        return self._cached_train_data, self._cached_test_data
    
    def _get_embeddings_if_needed(self, train_data: List[Dict]) -> Tuple[np.ndarray, Normalizer]:
        """Generate and cache embeddings."""
        if self._cached_embeddings is None or self._cached_normalizer is None:
            logger.info("Generating embeddings...")
            queries = [item["query"] for item in train_data]
            
            embeddings = self.embedder.batch(queries, max_batch_size=100)
            embeddings = np.array(embeddings)
            
            normalizer = Normalizer(norm="l2")
            normalized_embeddings = normalizer.fit_transform(embeddings)
            
            self._cached_embeddings = normalized_embeddings
            self._cached_normalizer = normalizer
            
            logger.info(f"Generated embeddings with shape: {normalized_embeddings.shape}")
        
        return self._cached_embeddings, self._cached_normalizer
    
    def _get_cluster(self, embeddings: np.ndarray, method_name: str, method_params: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Perform clustering with specified method and parameters."""
        if method_name == "kmeans":
            k = method_params.get('n_clusters', 64)
            kmeans = KMeans(
                n_clusters=k, random_state=self.seed,
                init="k-means++", n_init="auto",
                max_iter=1000, algorithm="elkan"
            )
            labels = kmeans.fit_predict(embeddings)
            centers = kmeans.cluster_centers_
            metric = kmeans.inertia_
            
        elif method_name == "hierarchical":
            k = method_params.get('n_clusters', 64)
            agg_model = AgglomerativeClustering(
                n_clusters=k, metric='euclidean', linkage='ward'
            )
            labels = agg_model.fit_predict(embeddings)
            unique_labels = np.unique(labels)
            centers = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
            metric = None
            
        elif method_name == "gmm":
            k = method_params.get('n_components', 64)
            gmm_model = GaussianMixture(
                n_components=k, covariance_type=method_params.get('covariance_type', 'full'),
                random_state=self.seed, n_init=10, max_iter=100
            )
            labels = gmm_model.fit_predict(embeddings)
            centers = gmm_model.means_
            metric = gmm_model.bic(embeddings)
            
        elif method_name == "birch":
            k = method_params.get('n_clusters', 64)
            birch_model = Birch(
                threshold=method_params.get('threshold', 0.5),
                branching_factor=method_params.get('branching_factor', 50),
                n_clusters=k
            )
            labels = birch_model.fit_predict(embeddings)
            unique_labels = np.unique(labels)
            centers = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
            metric = None
            
        elif method_name == "spectral":
            k = method_params.get('n_clusters', 64)
            spectral_model = SpectralClustering(
                n_clusters=k, affinity='rbf', gamma=0.1, random_state=self.seed
            )
            labels = spectral_model.fit_predict(embeddings)
            unique_labels = np.unique(labels)
            centers = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
            metric = None
            
        else:
            raise ValueError(f"Unknown clustering method: {method_name}")
        
        return labels, centers, metric
    
    def _compute_cluster_rankings(self, labels: np.ndarray, train_data: List[Dict]) -> Dict[int, Dict]:
        """Compute model rankings within each cluster."""
        cluster_stats = {}
        
        for cluster_id, sample in zip(labels, train_data):
            cluster_id = int(cluster_id)
            
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {'total': 0, 'scores': {}}
            
            cluster_stats[cluster_id]['total'] += 1
            
            for model_name, is_correct in sample['records'].items():
                if model_name not in cluster_stats[cluster_id]['scores']:
                    cluster_stats[cluster_id]['scores'][model_name] = {
                        'correct': 0, 'total': 0, 'accuracy': 0
                    }
                
                cluster_stats[cluster_id]['scores'][model_name]['total'] += 1
                if is_correct:
                    cluster_stats[cluster_id]['scores'][model_name]['correct'] += 1
        
        # Compute rankings
        rankings = {}
        for cluster_id, stats in cluster_stats.items():
            for model_name in stats['scores']:
                model_stats = stats['scores'][model_name]
                model_stats['accuracy'] = (
                    model_stats['correct'] / model_stats['total'] if model_stats['total'] > 0 else 0
                )
            
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
        
        return rankings
    
    def _route_models_batched(
        self, 
        queries: List[str],
        normalizer: Normalizer,
        centers: np.ndarray,
        rankings: Dict[int, Dict],
        top_n: int = 1,
        top_k: int = 1
    ) -> List[List[str]]:
        """Route queries to models using cluster-based ranking."""
        # Get query embeddings
        q_vecs = np.array(self.embedder.batch(queries))
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        q_vecs_normalized = normalizer.transform(q_vecs)
        
        # Calculate distances to all centers
        dists_matrix = 1 - centers @ q_vecs_normalized.T
        
        all_ranked_lists = []
        
        for i in range(q_vecs_normalized.shape[0]):
            query_dists = dists_matrix[:, i]
            idx = np.argsort(query_dists)[:top_k]
            sel_dists = query_dists[idx]
            
            # Convert distances to probabilities
            logits = -self.BETA * sel_dists
            P_exp = np.exp(logits - np.max(logits))
            P_sum = P_exp.sum()
            
            if P_sum <= 1e-9:
                P = np.ones_like(sel_dists) / len(sel_dists) if len(sel_dists) > 0 else np.array([])
            else:
                P = P_exp / P_sum
            
            # Compute model scores
            scores = {model: 0.0 for model in self.available_models}
            
            for cluster_idx, p_value in zip(idx, P):
                cluster_id = int(cluster_idx)
                
                if cluster_id not in rankings:
                    continue
                
                ranking = rankings[cluster_id].get('ranking', [])
                
                for model in self.available_models:
                    try:
                        rank = ranking.index(model)
                        score_component = 1.0 / (rank + 0.1)
                    except ValueError:
                        score_component = 1.0 / self.DEFAULT_RANK
                    
                    scores[model] += p_value * score_component
            
            ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            all_ranked_lists.append([m for m, _ in ranked_models[:top_n]])
        
        return all_ranked_lists
    
    def _evaluate_task_performance(
        self,
        data: List[Dict],
        centers: np.ndarray,
        rankings: Dict[int, Dict],
        normalizer: Normalizer,
        task: str,
        top_n: int = 1,
        top_k: int = 1
    ) -> float:
        """Evaluate performance on a specific task."""
        task_items = [item for item in data if item["dataset"] == task]
        if not task_items:
            return 0.0
        
        queries = [item["query"] for item in task_items]
        all_ranked_lists = self._route_models_batched(
            queries, normalizer, centers, rankings, top_n, top_k
        )
        
        correct_count = 0
        for i, item in enumerate(task_items):
            ranked_models = all_ranked_lists[i]
            if not ranked_models:
                continue
            
            records = item["records"]
            frequency_data = item.get("frequency", {})
            
            selected_model = ranked_models[0]
            is_correct = records.get(selected_model, False)
            model_frequency = frequency_data.get(selected_model, 0.0)
            
            # Use second model if first model has low confidence and top_n > 1
            if (top_n > 1 and len(ranked_models) > 1 and model_frequency < 1.0):
                second_model = ranked_models[1]
                second_correct = records.get(second_model, False)
                second_frequency = frequency_data.get(second_model, 0.0)
                
                if second_frequency > model_frequency:
                    correct_count += 1 if second_correct else 0
                else:
                    correct_count += 1 if is_correct else 0
            else:
                correct_count += 1 if is_correct else 0
        
        return 100 * correct_count / len(task_items)
    
    def _evaluate_performance(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        centers: np.ndarray,
        rankings: Dict[int, Dict],
        normalizer: Normalizer,
        top_n: int = 1,
        top_k: int = 1
    ) -> List[Dict]:
        """Evaluate performance across all datasets."""
        results = []
        
        def process_task(task):
            return {
                "dataset": task,
                "train": self._evaluate_task_performance(
                    train_data, centers, rankings, normalizer, task, top_n, top_k
                ),
                "test": self._evaluate_task_performance(
                    test_data, centers, rankings, normalizer, task, top_n, top_k
                )
            }
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_task, task) for task in self.dataset_names]
            for future in as_completed(futures):
                results.append(future.result())
        
        return results
    
    def experiment_k_sensitivity(
        self,
        k_range: List[int],
        output_dir: str = "experiments/k_sensitivity",
        top_k: int = 1
    ) -> pd.DataFrame:
        """
        Experiment 1: K-means cluster number sensitivity analysis.
        
        Args:
            k_range: List of K values to test
            output_dir: Directory to save results
            top_k: Number of top clusters to consider for routing
            
        Returns:
            DataFrame with results for each K value
        """
        logger.info(f"Starting K sensitivity experiment with K values: {k_range}")
        
        os.makedirs(output_dir, exist_ok=True)
        train_data, test_data = self._load_data_if_needed()
        embeddings, normalizer = self._get_embeddings_if_needed(train_data)
        
        all_results = []
        
        for k in tqdm(k_range, desc="K sensitivity"):
            result_file = Path(output_dir) / f"k_{k}.csv"
            if result_file.exists():
                logger.info(f"K={k} result exists, loading from {result_file}")
                df = pd.read_csv(result_file)
                all_results.append(df)
                continue
            
            try:
                # Perform clustering
                labels, centers, inertia = self._get_cluster(
                    embeddings, "kmeans", {"n_clusters": k}
                )
                
                # Compute rankings
                rankings = self._compute_cluster_rankings(labels, train_data)
                
                # Evaluate performance
                performance = self._evaluate_performance(
                    train_data, test_data, centers, rankings, normalizer, top_n=1, top_k=top_k
                )
                
                # Format results
                results_df = pd.DataFrame(performance)
                results_df['k'] = k
                results_df['inertia'] = inertia
                results_df['num_clusters_found'] = len(np.unique(labels))
                
                # Save results
                results_df.to_csv(result_file, index=False)
                all_results.append(results_df)
                
                logger.info(f"K={k} completed, inertia={inertia:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing K={k}: {str(e)}")
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(Path(output_dir) / "k_sensitivity_combined.csv", index=False)
        
        logger.info(f"K sensitivity experiment completed. Results saved to {output_dir}")
        return combined_df
    
    def experiment_clustering_methods(
        self,
        methods_config: Dict[str, Dict],
        output_dir: str = "experiments/clustering_methods",
        fixed_k: int = 64
    ) -> pd.DataFrame:
        """
        Experiment 2: Compare different clustering methods.
        
        Args:
            methods_config: Dict mapping method names to their parameters
            output_dir: Directory to save results  
            fixed_k: Fixed K value for methods that use it
            
        Returns:
            DataFrame comparing all clustering methods
        """
        logger.info(f"Starting clustering methods comparison with K={fixed_k}")
        
        os.makedirs(output_dir, exist_ok=True)
        train_data, test_data = self._load_data_if_needed()
        embeddings, normalizer = self._get_embeddings_if_needed(train_data)
        
        if not methods_config:
            # Default methods configuration
            methods_config = {
                "kmeans": {"n_clusters": fixed_k},
                "hierarchical": {"n_clusters": fixed_k},
                "gmm": {"n_components": fixed_k, "covariance_type": "diag"},
                "birch": {"n_clusters": fixed_k, "threshold": 0.5},
                "spectral": {"n_clusters": fixed_k}
            }
        
        all_results = []
        
        for method_name, params in tqdm(methods_config.items(), desc="Clustering methods"):
            result_file = Path(output_dir) / f"method_{method_name}.csv"
            if result_file.exists():
                logger.info(f"Method {method_name} result exists, loading")
                df = pd.read_csv(result_file)
                all_results.append(df)
                continue
            
            try:
                # Perform clustering
                labels, centers, metric = self._get_cluster(embeddings, method_name, params)
                
                if centers.shape[0] == 0:
                    logger.warning(f"Method {method_name} produced no clusters, skipping")
                    continue
                
                # Compute rankings
                rankings = self._compute_cluster_rankings(labels, train_data)
                
                # Evaluate performance
                performance = self._evaluate_performance(
                    train_data, test_data, centers, rankings, normalizer
                )
                
                # Format results
                results_df = pd.DataFrame(performance)
                results_df['method'] = method_name
                results_df['num_clusters'] = centers.shape[0]
                results_df['metric_value'] = metric
                results_df['params'] = str(params)
                
                # Save results
                results_df.to_csv(result_file, index=False)
                all_results.append(results_df)
                
                logger.info(f"Method {method_name} completed, clusters={centers.shape[0]}")
                
            except Exception as e:
                logger.error(f"Error processing method {method_name}: {str(e)}")
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(Path(output_dir) / "clustering_methods_combined.csv", index=False)
        
        logger.info(f"Clustering methods experiment completed. Results saved to {output_dir}")
        return combined_df
    
    def experiment_top_k_analysis(
        self,
        top_k_range: List[int],
        output_dir: str = "experiments/top_k_analysis",
        fixed_k: int = 64
    ) -> pd.DataFrame:
        """
        Experiment 3: Analyze impact of top-K cluster selection.
        
        Args:
            top_k_range: List of top-K values to test
            output_dir: Directory to save results
            fixed_k: Fixed number of clusters for K-means
            
        Returns:
            DataFrame with results for each top-K value
        """
        logger.info(f"Starting top-K analysis with values: {top_k_range}")
        
        os.makedirs(output_dir, exist_ok=True)
        train_data, test_data = self._load_data_if_needed()
        embeddings, normalizer = self._get_embeddings_if_needed(train_data)
        
        # Perform clustering once with fixed K
        labels, centers, inertia = self._get_cluster(embeddings, "kmeans", {"n_clusters": fixed_k})
        rankings = self._compute_cluster_rankings(labels, train_data)
        
        all_results = []
        
        for top_k in tqdm(top_k_range, desc="Top-K analysis"):
            try:
                # Evaluate with different top_k values
                performance = self._evaluate_performance(
                    train_data, test_data, centers, rankings, normalizer, top_n=1, top_k=top_k
                )
                
                # Format results
                results_df = pd.DataFrame(performance)
                results_df['top_k'] = top_k
                results_df['fixed_k'] = fixed_k
                results_df['inertia'] = inertia
                
                all_results.append(results_df)
                logger.info(f"Top-K={top_k} completed")
                
            except Exception as e:
                logger.error(f"Error processing top-K={top_k}: {str(e)}")
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(Path(output_dir) / "top_k_analysis_combined.csv", index=False)
        
        logger.info(f"Top-K analysis experiment completed. Results saved to {output_dir}")
        return combined_df
    
    def experiment_test_size_analysis(
        self,
        test_size_range: List[float],
        output_dir: str = "experiments/test_size_analysis",
        fixed_k: int = 64
    ) -> pd.DataFrame:
        """
        Experiment 4: Analyze impact of train/test split ratio.
        
        Args:
            test_size_range: List of test size ratios to test
            output_dir: Directory to save results
            fixed_k: Fixed number of clusters for K-means
            
        Returns:
            DataFrame with results for each test size ratio
        """
        logger.info(f"Starting test size analysis with ratios: {test_size_range}")
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        
        for test_size in tqdm(test_size_range, desc="Test size analysis"):
            try:
                # Clear cached data to force reload with new split
                self._cached_train_data = None
                self._cached_test_data = None
                self._cached_embeddings = None
                self._cached_normalizer = None
                
                # Load data with new split
                train_data, test_data = self._load_data_if_needed(test_size=test_size)
                embeddings, normalizer = self._get_embeddings_if_needed(train_data)
                
                # Perform clustering
                labels, centers, inertia = self._get_cluster(embeddings, "kmeans", {"n_clusters": fixed_k})
                rankings = self._compute_cluster_rankings(labels, train_data)
                
                # Evaluate performance
                performance = self._evaluate_performance(
                    train_data, test_data, centers, rankings, normalizer
                )
                
                # Format results
                results_df = pd.DataFrame(performance)
                results_df['test_size'] = test_size
                results_df['train_size'] = len(train_data)
                results_df['test_size_count'] = len(test_data)
                results_df['fixed_k'] = fixed_k
                results_df['inertia'] = inertia
                
                all_results.append(results_df)
                logger.info(f"Test size={test_size} completed, train={len(train_data)}, test={len(test_data)}")
                
            except Exception as e:
                logger.error(f"Error processing test_size={test_size}: {str(e)}")
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(Path(output_dir) / "test_size_analysis_combined.csv", index=False)
        
        logger.info(f"Test size analysis experiment completed. Results saved to {output_dir}")
        return combined_df


def main():
    """Command line interface for ablation experiments."""
    parser = argparse.ArgumentParser(description="Run ablation experiments for Avengers rank router")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["k_sensitivity", "clustering_methods", "top_k_analysis", "test_size_analysis", "all"],
                        help="Experiment type to run")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Base output directory")
    parser.add_argument("--embed_model", type=str, default="gte-qwen2-7b-instruct", help="Embedding model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory")
    
    # Experiment-specific arguments
    parser.add_argument("--k_range", type=str, default="1,10,20,30,40,50,64,80,100,150,200", 
                        help="Comma-separated K values for k_sensitivity")
    parser.add_argument("--top_k_range", type=str, default="1,2,3,4,5,10", 
                        help="Comma-separated top-K values for top_k_analysis")
    parser.add_argument("--test_size_range", type=str, default="0.1,0.2,0.3,0.4,0.5", 
                        help="Comma-separated test size ratios for test_size_analysis")
    parser.add_argument("--fixed_k", type=int, default=64, help="Fixed K for experiments that need it")
    
    args = parser.parse_args()
    
    # Default configuration
    available_models = ['M01', 'M02', 'M04', 'M05', 'M09', 'M10', 'M11', 'M21', 'M22', 'M16']
    dataset_names = [
        "aime", "math500", "livemathbench", "emorynlp", "meld",
        "korbench", "bbh", "kandk", "mmlupro", "gpqa", 
        "arcc", "finqa", "medqa", "humaneval", "mbpp"
    ]
    
    embed_config = EMBED_CONFIG.get(args.embed_model)
    if not embed_config:
        raise ValueError(f"Unknown embedding model: {args.embed_model}")
    
    # Initialize runner
    runner = AblationExperimentRunner(
        embed_config=embed_config,
        data_path=args.data_path,
        available_models=available_models,
        dataset_names=dataset_names,
        cache_dir=args.cache_dir,
        seed=args.seed
    )
    
    # Run experiments
    if args.experiment == "k_sensitivity" or args.experiment == "all":
        k_range = [int(x) for x in args.k_range.split(",")]
        runner.experiment_k_sensitivity(k_range, f"{args.output_dir}/k_sensitivity")
    
    if args.experiment == "clustering_methods" or args.experiment == "all":
        runner.experiment_clustering_methods({}, f"{args.output_dir}/clustering_methods", args.fixed_k)
    
    if args.experiment == "top_k_analysis" or args.experiment == "all":
        top_k_range = [int(x) for x in args.top_k_range.split(",")]
        runner.experiment_top_k_analysis(top_k_range, f"{args.output_dir}/top_k_analysis", args.fixed_k)
    
    if args.experiment == "test_size_analysis" or args.experiment == "all":
        test_size_range = [float(x) for x in args.test_size_range.split(",")]
        runner.experiment_test_size_analysis(test_size_range, f"{args.output_dir}/test_size_analysis", args.fixed_k)
    
    logger.info("All requested experiments completed!")


if __name__ == "__main__":
    main()