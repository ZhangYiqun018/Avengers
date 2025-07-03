# Usage Guide

This guide provides step-by-step instructions for using the Avengers framework to evaluate and ensemble multiple language models.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ZhangYiqun018/Avengers.git
cd Avengers

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Configuration

Create a configuration file by copying the template:

```bash
cp config/experts_template.yaml config/my_config.yaml
```

Edit `config/my_config.yaml` to configure your models and API endpoints.

### 3. Run a Simple Experiment

```bash
python app.py --config config/my_config.yaml --save_dir results/
```

## Configuration

### Basic Configuration Structure

```yaml
experiments:
  task: "humaneval"          # Task to evaluate
  max_workers: 4             # Parallel workers
  mode: "test"               # "test" or "full"
  use_http_cache: true       # Enable caching
  cache_dir: "cache"         # Cache directory

router:
  type: "straight"           # Router type
  straight_router:
    model: "model_name"      # Model to use

generator:
  type: "direct"             # Generator type
  direct:
    temperature: 0.2
    top_p: 1.0

experts:
  - name: "model1"
    base_url: "http://your-api-endpoint/v1"
    api_key: "your-api-key"
    description: "Model 1 description"
  
  - name: "model2"
    base_url: "http://your-api-endpoint/v1"
    api_key: "your-api-key"
    description: "Model 2 description"
```

### Available Tasks

| Category | Tasks |
|----------|-------|
| **Mathematics** | `aime`, `math500`, `livemathbench`, `mathbench` |
| **Code** | `humaneval`, `mbpp`, `livecodebench`, `studenteval` |
| **Logic** | `kandk`, `bbh`, `korbench`, `brainteaser` |
| **Knowledge** | `finqa`, `medqa`, `gpqa`, `arcc`, `mmlupro` |
| **Affective** | `emorynlp`, `meld`, `dailydialog` |

### Router Types

#### 1. Straight Router (Single Model)
```yaml
router:
  type: "straight"
  straight_router:
    model: "your_model_name"
```

#### 2. Random Router (Random Selection)
```yaml
router:
  type: "random"
  random_router:
    max_router: 2
```

#### 3. Rank Router (Cluster-Based Selection)
```yaml
router:
  type: "rank"
  rank_router:
    centres_path: "path/to/centres.npy"
    rankings_path: "path/to/rankings.json"
    mapping_path: "path/to/mapping.json"
    normalizer_path: "path/to/normalizer.joblib"
    available_models:
      - "model1"
      - "model2"
    top_n: 2
    top_k: 2
    beta: 6.0
    embedding_model: "your_embedding_model"
```

#### 4. GPT Router (LLM-Based Routing)
```yaml
router:
  type: "gpt"
  gpt_router:
    model: "gpt-4"
    max_router: 2
    base_url: "https://api.openai.com/v1"
    api_key: "your-openai-api-key"
```

### Generator Types

#### 1. Direct Generator (Single Inference)
```yaml
generator:
  type: "direct"
  direct:
    temperature: 0.2
    top_p: 1.0
```

#### 2. Self-Consistency Generator (Multiple Samples)
```yaml
generator:
  type: "self_consistency"
  self_consistency:
    samples: 10
    temperature: 0.7
    top_p: 1.0
```

#### 3. Fast-Slow Generator (Two-Stage Generation)
```yaml
generator:
  type: "fast_slow"
  fast_slow:
    fast_samples: 10
    fast_temperature: 0.7
    fast_top_p: 1.0
    slow_samples: 1
    slow_temperature: 0.7
    slow_top_p: 1.0
    consistency_rate_threshold: 0.8
```

## Running Experiments

### Single Experiment

```bash
python app.py --config config/my_config.yaml --save_dir results/
```

### Multiple Experiments

Create an experiment variations file:

```yaml
# config/experiment_variations.yaml
base_config: config/experts.yaml
save_dir: results/
variations:
  - name: "humaneval_straight"
    params:
      experiments.task: "humaneval"
      router.type: "straight"
      router.straight_router.model: "model1"
      generator.type: "direct"

  - name: "humaneval_random"
    params:
      experiments.task: "humaneval"
      router.type: "random"
      router.random_router.max_router: 2
      generator.type: "self_consistency"
      generator.self_consistency.samples: 5
```

Run multiple experiments:

```bash
python main.py --config config/experiment_variations.yaml
```

## Common Use Cases

### 1. Evaluate Single Model
```yaml
experiments:
  task: "humaneval"
  max_workers: 4
  mode: "test"

router:
  type: "straight"
  straight_router:
    model: "your_model"

generator:
  type: "direct"
  direct:
    temperature: 0.2

experts:
  - name: "your_model"
    base_url: "http://your-endpoint/v1"
    api_key: "your-key"
```

### 2. Compare Multiple Models
```yaml
experiments:
  task: "humaneval"
  max_workers: 4
  mode: "test"

router:
  type: "random"
  random_router:
    max_router: 3

generator:
  type: "self_consistency"
  self_consistency:
    samples: 5
    temperature: 0.7

experts:
  - name: "model1"
    base_url: "http://endpoint1/v1"
    api_key: "key1"
  - name: "model2"
    base_url: "http://endpoint2/v1"
    api_key: "key2"
  - name: "model3"
    base_url: "http://endpoint3/v1"
    api_key: "key3"
```

### 3. Use The Avengers Method (Cluster-Based Routing)
```yaml
experiments:
  task: "humaneval"
  max_workers: 4
  mode: "test"

router:
  type: "rank"
  rank_router:
    centres_path: "path/to/centres.npy"
    rankings_path: "path/to/rankings.json"
    mapping_path: "path/to/mapping.json"
    normalizer_path: "path/to/normalizer.joblib"
    available_models:
      - "model1"
      - "model2"
      - "model3"
    top_n: 3
    top_k: 2
    beta: 6.0
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

generator:
  type: "self_consistency"
  self_consistency:
    samples: 10
    temperature: 0.7

experts:
  - name: "model1"
    base_url: "http://endpoint1/v1"
    api_key: "key1"
  - name: "model2"
    base_url: "http://endpoint2/v1"
    api_key: "key2"
  - name: "model3"
    base_url: "http://endpoint3/v1"
    api_key: "key3"
```

## Performance Optimization

### 1. Enable HTTP Caching
```yaml
experiments:
  use_http_cache: true
  cache_dir: "cache"
```

### 2. Adjust Worker Count
```yaml
experiments:
  max_workers: 8  # Adjust based on your system
```

### 3. Use Test Mode for Quick Evaluation
```yaml
experiments:
  mode: "test"  # Use "full" for complete evaluation
```

## Result Analysis

Results are saved as JSON files in the specified directory with the following structure:

```
results/
├── direct/
│   ├── humaneval-20231201-143022.json
│   └── math500-20231201-143523.json
├── self_consistency/
│   ├── humaneval-20231201-144015.json
│   └── ...
```

Each result file contains:
- Configuration used
- Per-sample results
- Aggregate metrics
- Timing information

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all API keys are correctly configured
2. **Model Not Found**: Check that model names match your API endpoints
3. **Memory Issues**: Reduce `max_workers` if running out of memory
4. **Slow Performance**: Enable HTTP caching and use test mode

### Debug Mode

Add logging configuration to see detailed execution:

```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

## Advanced Usage

### Custom Evaluation Tasks

To add new evaluation tasks, create a new evaluator in the `evaluate/` directory following the existing patterns.

### Custom Routers

Implement custom routing logic by extending the `BaseRouter` class in `core/routing/`.

### Custom Generators

Create custom generation strategies by extending the `BaseGenerator` class in `core/inference/`.

For more advanced usage patterns, refer to the source code documentation and examples in the repository.