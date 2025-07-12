# Rank Router Implementation

This directory contains the implementation of the rank-based routing system for the Avengers framework, including automated generation tools and comprehensive ablation studies.

## File Structure

```
core/
├── generate_rank_router.py     # Unified rank router generation script
├── ablation_experiments.py     # Unified ablation experiment framework
├── ablation/                   # Original ablation experiment code
│   ├── embedding_cache.py      # Embedding vector cache
│   ├── utils.py               # Configuration and utility functions
│   ├── explore_k.py           # K-value sensitivity analysis
│   ├── explore_cluster_method.py  # Clustering method comparison
│   ├── explore_select_model_method.py  # Model selection strategies
│   ├── explore_test_size.py   # Train/test split analysis
│   └── explore_kmeans.py      # K-means specific experiments
└── rank/                      # Generated rank router files
    ├── ranking_*.json         # Cluster ranking files
    ├── ranking_centers_*.npy  # Cluster centers
    ├── ranking_*_normalizer.joblib  # Normalizers
    └── map_*.json            # Model mapping files
```

## 1. Rank Router Generation (`generate_rank_router.py`)

### Overview

The rank router generation script provides a complete pipeline for creating cluster-based routing artifacts from training data. The system:

1. **Data loading and preprocessing**: Load training data, stratified train/test split by dataset
2. **Embedding generation**: Generate vector representations using specified embedding models
3. **Clustering analysis**: Cluster queries using K-means
4. **Ranking computation**: Compute model performance rankings within each cluster
5. **Model selection**: Automatically select optimal model combinations based on cluster rankings
6. **File generation**: Generate all configuration files needed for rank router

### Usage

#### Basic Usage

```bash
python core/generate_rank_router.py \
    --data_path /path/to/your/training_data.json \
    --output_dir core/rank \
    --embed_model gte-qwen2-7b-instruct \
    --n_clusters 64 \
    --n_models 10
```

#### Full Parameters

```bash
python core/generate_rank_router.py \
    --data_path data/training_data.json \
    --output_dir core/rank \
    --embed_model gte-qwen2-7b-instruct \
    --embed_url http://your-embedding-api:8000/v1 \
    --embed_api_key your-api-key \
    --n_clusters 64 \
    --n_models 10 \
    --test_size 0.3 \
    --seed 42 \
    --cache_dir .cache
```

#### Programming Interface

```python
from core.generate_rank_router import RankRouterGenerator

# Configure embedding model
embed_config = {
    "url": "http://172.30.12.113:8000/v1",
    "model_name": "gte-qwen2-7b-instruct"
}

# Initialize generator
generator = RankRouterGenerator(
    embed_config=embed_config,
    cache_dir=".cache"
)

# Generate artifacts
artifacts = generator.generate_artifacts(
    data_path="data/training_data.json",
    output_dir="core/rank",
    n_clusters=64,
    n_selected_models=10
)

print("Generated files:", artifacts)
```

### Generated Files

After running, the following files will be generated in `output_dir`:

- `ranking_split_k64_m22.json`: Model rankings for each cluster
- `ranking_centers_split_k64_m22.npy`: Cluster centers
- `ranking_embedding_normalizer_gte_qwen2_7b_instruct.joblib`: Normalizer
- `map_m22.json`: Model name to ID mapping
- `selected_models_m10.json`: Automatically selected best model list

### Integration with Avengers Framework

To use the generated artifacts in your Avengers experiments, configure the rank router as follows:

```yaml
router:
  type: "rank"
  rank_router:
    centres_path: "core/rank/ranking_centers_split_k64_m22.npy"
    rankings_path: "core/rank/ranking_split_k64_m22.json"
    mapping_path: "core/rank/map_m22.json"
    normalizer_path: "core/rank/ranking_embedding_normalizer_gte_qwen2_7b_instruct.joblib"
    available_models:
      - "model1_name"
      - "model2_name"
      # ... Your selected model names
    top_n: 2
    top_k: 3
    beta: 6.0
    embedding_model: "gte-qwen2-7b-instruct"
```

## 2. Ablation Experiment Framework (`ablation_experiments.py`)

### Overview

Comprehensive ablation study framework for evaluating different components of the rank router system. The framework supports:

1. **K-value sensitivity analysis**: Test the impact of different cluster numbers on performance
2. **Clustering method comparison**: Compare K-means, hierarchical clustering, GMM, Birch, Spectral, etc.
3. **Top-K analysis**: Analyze the impact of considering top-K nearest clusters on routing performance
4. **Training set size analysis**: Study the impact of train/test split ratios

### Usage

#### Run Individual Experiments

```bash
# K-value sensitivity analysis
python core/ablation_experiments.py \
    --experiment k_sensitivity \
    --data_path data/training_data.json \
    --k_range "1,10,20,30,40,50,64,80,100,150,200" \
    --output_dir experiments

# Clustering method comparison
python core/ablation_experiments.py \
    --experiment clustering_methods \
    --data_path data/training_data.json \
    --fixed_k 64 \
    --output_dir experiments

# Top-K analysis
python core/ablation_experiments.py \
    --experiment top_k_analysis \
    --data_path data/training_data.json \
    --top_k_range "1,2,3,4,5,10" \
    --fixed_k 64 \
    --output_dir experiments

# Test size analysis
python core/ablation_experiments.py \
    --experiment test_size_analysis \
    --data_path data/training_data.json \
    --test_size_range "0.1,0.2,0.3,0.4,0.5" \
    --fixed_k 64 \
    --output_dir experiments
```

#### Run All Experiments

```bash
python core/ablation_experiments.py \
    --experiment all \
    --data_path data/training_data.json \
    --output_dir experiments \
    --embed_model gte-qwen2-7b-instruct \
    --seed 42
```

#### Programming Interface

```python
from core.ablation_experiments import AblationExperimentRunner

# Configuration
embed_config = {
    "url": "http://172.30.12.113:8000/v1",
    "model_name": "gte-qwen2-7b-instruct"
}

available_models = ['M01', 'M02', 'M04', 'M05', 'M09', 'M10', 'M11', 'M21', 'M22', 'M16']
dataset_names = ["aime", "math500", "humaneval", "mbpp", "gpqa", "mmlupro"]

# Initialize
runner = AblationExperimentRunner(
    embed_config=embed_config,
    data_path="data/training_data.json",
    available_models=available_models,
    dataset_names=dataset_names
)

# Run experiments
k_results = runner.experiment_k_sensitivity([1, 10, 20, 50, 64, 100])
method_results = runner.experiment_clustering_methods({}, fixed_k=64)
topk_results = runner.experiment_top_k_analysis([1, 2, 3, 5], fixed_k=64)
```

### Experiment Results

Each experiment generates CSV files containing:

- Train/test performance on each dataset
- Experiment parameters (K values, method names, etc.)
- Clustering quality metrics (e.g., inertia)
- Summary statistics

Result file structure:
```
experiments/
├── k_sensitivity/
│   ├── k_1.csv, k_10.csv, ..., k_200.csv
│   └── k_sensitivity_combined.csv
├── clustering_methods/
│   ├── method_kmeans.csv, method_hierarchical.csv, ...
│   └── clustering_methods_combined.csv
├── top_k_analysis/
│   └── top_k_analysis_combined.csv
└── test_size_analysis/
    └── test_size_analysis_combined.csv
```

## 3. Key Improvements

### Improvements Over Original Code

1. **Unified interface**: All functionality accessible through unified classes and command-line interfaces
2. **Code reuse**: Eliminated extensive code duplication, shared core functionality
3. **Error handling**: Better exception handling and logging
4. **Parallelization**: Support for multi-threaded parallel evaluation of different datasets
5. **Cache optimization**: Smart caching of embedding vectors, avoiding redundant computation
6. **Configuration flexibility**: Support for custom embedding models, API endpoints, parameters, etc.
7. **Result management**: Automatic saving of intermediate results, support for resuming from checkpoints

### Preserved Core Logic

1. **Clustering algorithms**: Maintained original K-means and other clustering method implementations
2. **Ranking computation**: Maintained original accuracy-based model ranking logic  
3. **Routing strategy**: Maintained original distance and ranking-based routing algorithm
4. **Evaluation metrics**: Maintained original performance evaluation methods

## 4. Usage Recommendations

### Typical Workflow

1. **First-time use**:
   ```bash
   # Generate rank router files
   python core/generate_rank_router.py --data_path your_data.json
   
   # Run ablation experiments to validate performance
   python core/ablation_experiments.py --experiment all --data_path your_data.json
   ```

2. **Parameter tuning**:
   ```bash
   # Test different K values
   python core/ablation_experiments.py --experiment k_sensitivity --k_range "32,64,128,256"
   
   # Regenerate files with optimal K value
   python core/generate_rank_router.py --n_clusters 128
   ```

3. **Deployment**:
   - Copy generated files to `core/rank/` directory
   - Reference these files in configuration
   - Run your Avengers experiments

### Performance Optimization Tips

1. **Embedding vector caching**: First run caches embedding vectors, subsequent experiments reuse them
2. **Parallel processing**: Adjust `max_workers` parameter to balance speed and resource usage
3. **Batch size**: Adjust `max_batch_size` according to API limits
4. **GPU acceleration**: If cuML is installed, K-means will automatically use GPU

## Contributing

We welcome contributions to improve the rank router implementation. Please feel free to:

- Submit bug reports and feature requests via GitHub issues
- Contribute new clustering methods or evaluation metrics
- Improve documentation and examples
- Add support for additional embedding models

## Citation

If you use this rank router implementation in your research, please cite:

```bibtex
@misc{zhang2025avengerssimplerecipeuniting,
      title={The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants}, 
      author={Yiqun Zhang and Hao Li and Chenxu Wang and Linyao Chen and Qiaosheng Zhang and Peng Ye and Shi Feng and Daling Wang and Zhen Wang and Xinrun Wang and Jia Xu and Lei Bai and Wanli Ouyang and Shuyue Hu},
      year={2025},
      eprint={2505.19797},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.19797}, 
}
```