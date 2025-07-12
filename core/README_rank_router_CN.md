# Rank Router 实现

本目录包含 Avengers 框架中基于排名的路由系统实现，包括自动化生成工具和全面的消融研究。

## 文件结构

```
core/
├── generate_rank_router.py     # 统一的 rank router 生成脚本
├── ablation_experiments.py     # 统一的消融实验框架
├── ablation/                   # 原始消融实验代码
│   ├── embedding_cache.py      # 嵌入向量缓存
│   ├── utils.py               # 配置和工具函数
│   ├── explore_k.py           # K值敏感性分析
│   ├── explore_cluster_method.py  # 聚类方法比较
│   ├── explore_select_model_method.py  # 模型选择策略
│   ├── explore_test_size.py   # 训练测试集划分分析
│   └── explore_kmeans.py      # K-means 特定实验
└── rank/                      # 生成的 rank router 文件
    ├── ranking_*.json         # 聚类排名文件
    ├── ranking_centers_*.npy  # 聚类中心
    ├── ranking_*_normalizer.joblib  # 归一化器
    └── map_*.json            # 模型映射文件
```

## 1. Rank Router 生成 (`generate_rank_router.py`)

### 功能概述

rank router 生成脚本提供了从训练数据创建基于聚类的路由配置文件的完整流水线。系统功能包括：

1. **数据加载和预处理**：加载训练数据，按数据集分层划分训练/测试集
2. **嵌入向量生成**：使用指定的嵌入模型生成查询的向量表示
3. **聚类分析**：使用 K-means 对查询进行聚类
4. **排名计算**：计算每个聚类中模型的性能排名
5. **模型选择**：基于聚类排名自动选择最佳模型组合
6. **文件生成**：生成 rank router 所需的所有配置文件

### 使用方法

#### 基本用法

```bash
python core/generate_rank_router.py \
    --data_path /path/to/your/training_data.json \
    --output_dir core/rank \
    --embed_model gte-qwen2-7b-instruct \
    --n_clusters 64 \
    --n_models 10
```

#### 完整参数

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

#### 编程接口

```python
from core.generate_rank_router import RankRouterGenerator

# 配置嵌入模型
embed_config = {
    "url": "http://172.30.12.113:8000/v1",
    "model_name": "gte-qwen2-7b-instruct"
}

# 初始化生成器
generator = RankRouterGenerator(
    embed_config=embed_config,
    cache_dir=".cache"
)

# 生成 artifacts
artifacts = generator.generate_artifacts(
    data_path="data/training_data.json",
    output_dir="core/rank",
    n_clusters=64,
    n_selected_models=10
)

print("Generated files:", artifacts)
```

### 生成的文件

运行后会在 `output_dir` 中生成以下文件：

- `ranking_split_k64_m22.json`：每个聚类的模型排名
- `ranking_centers_split_k64_m22.npy`：聚类中心
- `ranking_embedding_normalizer_gte_qwen2_7b_instruct.joblib`：归一化器
- `map_m22.json`：模型名称到ID的映射
- `selected_models_m10.json`：自动选择的最佳模型列表

### 与 Avengers 框架集成

要在 Avengers 实验中使用生成的配置文件，请按如下方式配置 rank router：

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
      # ... 您选择的模型名称
    top_n: 2
    top_k: 3
    beta: 6.0
    embedding_model: "gte-qwen2-7b-instruct"
```

## 2. 消融实验框架 (`ablation_experiments.py`)

### 功能概述

全面的消融研究框架，用于评估 rank router 系统的不同组件。框架支持：

1. **K值敏感性分析**：测试不同聚类数量对性能的影响
2. **聚类方法比较**：比较 K-means、层次聚类、GMM、Birch、Spectral 等方法
3. **Top-K 分析**：分析考虑前K个最近聚类对路由性能的影响
4. **训练集大小分析**：研究训练/测试集划分比例的影响

### 使用方法

#### 运行单个实验

```bash
# K值敏感性分析
python core/ablation_experiments.py \
    --experiment k_sensitivity \
    --data_path data/training_data.json \
    --k_range "1,10,20,30,40,50,64,80,100,150,200" \
    --output_dir experiments

# 聚类方法比较
python core/ablation_experiments.py \
    --experiment clustering_methods \
    --data_path data/training_data.json \
    --fixed_k 64 \
    --output_dir experiments

# Top-K 分析
python core/ablation_experiments.py \
    --experiment top_k_analysis \
    --data_path data/training_data.json \
    --top_k_range "1,2,3,4,5,10" \
    --fixed_k 64 \
    --output_dir experiments

# 测试集大小分析
python core/ablation_experiments.py \
    --experiment test_size_analysis \
    --data_path data/training_data.json \
    --test_size_range "0.1,0.2,0.3,0.4,0.5" \
    --fixed_k 64 \
    --output_dir experiments
```

#### 运行所有实验

```bash
python core/ablation_experiments.py \
    --experiment all \
    --data_path data/training_data.json \
    --output_dir experiments \
    --embed_model gte-qwen2-7b-instruct \
    --seed 42
```

#### 编程接口

```python
from core.ablation_experiments import AblationExperimentRunner

# 配置
embed_config = {
    "url": "http://172.30.12.113:8000/v1",
    "model_name": "gte-qwen2-7b-instruct"
}

available_models = ['M01', 'M02', 'M04', 'M05', 'M09', 'M10', 'M11', 'M21', 'M22', 'M16']
dataset_names = ["aime", "math500", "humaneval", "mbpp", "gpqa", "mmlupro"]

# 初始化
runner = AblationExperimentRunner(
    embed_config=embed_config,
    data_path="data/training_data.json",
    available_models=available_models,
    dataset_names=dataset_names
)

# 运行实验
k_results = runner.experiment_k_sensitivity([1, 10, 20, 50, 64, 100])
method_results = runner.experiment_clustering_methods({}, fixed_k=64)
topk_results = runner.experiment_top_k_analysis([1, 2, 3, 5], fixed_k=64)
```

### 实验结果

每个实验会生成 CSV 文件，包含：

- 各数据集的训练/测试性能
- 实验参数（K值、方法名、等）
- 聚类质量指标（如 inertia）
- 汇总统计

结果文件结构：
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

## 3. 主要改进

### 相比原始代码的改进

1. **统一接口**：所有功能通过统一的类和命令行接口访问
2. **代码复用**：消除了大量重复代码，共享核心功能
3. **错误处理**：更好的异常处理和日志记录
4. **并行化**：支持多线程并行评估不同数据集
5. **缓存优化**：智能缓存嵌入向量，避免重复计算
6. **配置灵活性**：支持自定义嵌入模型、API端点、参数等
7. **结果管理**：自动保存中间结果，支持断点续跑

### 保持的核心逻辑

1. **聚类算法**：保持原有的 K-means 和其他聚类方法实现
2. **排名计算**：保持原有的基于准确率的模型排名逻辑  
3. **路由策略**：保持原有的基于距离和排名的路由算法
4. **评估指标**：保持原有的性能评估方式

## 4. 使用建议

### 典型工作流程

1. **首次使用**：
   ```bash
   # 生成 rank router 文件
   python core/generate_rank_router.py --data_path your_data.json
   
   # 运行消融实验验证性能
   python core/ablation_experiments.py --experiment all --data_path your_data.json
   ```

2. **调优参数**：
   ```bash
   # 测试不同K值
   python core/ablation_experiments.py --experiment k_sensitivity --k_range "32,64,128,256"
   
   # 重新生成最佳K值的文件
   python core/generate_rank_router.py --n_clusters 128
   ```

3. **部署使用**：
   - 将生成的文件复制到 `core/rank/` 目录
   - 在配置文件中引用这些文件
   - 运行您的 Avengers 实验

### 性能优化建议

1. **嵌入向量缓存**：首次运行会缓存嵌入向量，后续实验会重用
2. **并行处理**：调整 `max_workers` 参数来平衡速度和资源使用
3. **批处理大小**：根据API限制调整 `max_batch_size`
4. **GPU加速**：如果安装了 cuML，K-means 会自动使用GPU

## 贡献指南

我们欢迎社区贡献来改进 rank router 实现。您可以：

- 通过 GitHub Issues 提交错误报告和功能请求
- 贡献新的聚类方法或评估指标
- 改进文档和示例
- 添加对更多嵌入模型的支持

## 引用

如果您在研究中使用了此 rank router 实现，请引用：

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