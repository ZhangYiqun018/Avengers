<div align="center">
<img src="./assets/overall.png" alt="Method Overview" width="900">
</div>

<h1 align="center">The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-6479326-b31b1b.svg)](https://arxiv.org/abs/6479326)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

</div>


## Abstract

The *Avengers* is a simple recipe that effectively leverages the *collective intelligence* of open-source, smaller language models. As of May 2025, with 10 open-source models (～7B parameters each), the *Avengers* collectively outperforms GPT-4.1 (OpenAI’s flagship model released on April 14, 2025) on 10 out of 15 representative datasets.

## Highlights

- No neural network training, no hand-crafted prompts or system architecture design
- Require only 1 parameter: the number of clusters
- Support automatic model selection and on-the-fly new model integration
- Robust to various embedding models, clustering algorithms, and ensemble strategies
- Comprehensive evaluation across 15 benchmarks covering 5 domains

<div align="center">
<img src="./assets/table1.png" alt="Main Results" width="850">
<p><em><strong>Table 1:</strong> Methodological comparison between baseline methods and the Avengers</em></p>
</div>

**Key advantages of The Avengers**: Unlike other ensemble methods, The Avengers requires **no training**, introduces **no additional parameters**, eliminates the need for designing **model collaboration structures**, and requires **no specialized prompt engineering**. This makes our approach uniquely lightweight and practical for real-world deployment.

## Results

<div align="center">
<img src="./assets/table2.png" alt="Main Result" width="850">
<p><em><strong>Table 2:</strong> Main Results</em></p>
</div>

**Outstanding performance**: Using approximately **10 models with ~7B parameters each**, The Avengers achieves superior performance compared to GPT-4.1 on **10 out of 15 datasets**, while also surpassing GPT-4.1 in **average performance across all 15 datasets**. When compared with baseline router- and mixture-based methods, The Avengers consistently delivers the best results across benchmarks.

<div align="center">
<img src="./assets/table3.png" alt="OOD Results" width="850">
<p><em><strong>Table 3:</strong> Out-of-distribution results</em></p>
</div>

**Superior generalization**: Compared to routing-based methods, The Avengers demonstrates the **best out-of-distribution (OOD) performance** thanks to its training-free nature, which prevents overfitting to specific datasets and ensures robust generalization to unseen domains.

<div align="center">
<img src="./assets/table4.png" alt="Ablation Study" width="850">
<p><em><strong>Table 4-6:</strong> Ablation studies.</em></p>
</div>

<div align="center">
<table border="0">
<tr>
    <td>
        <img src="./assets/figure2.png" alt="ablation" width="420">
        <p><em><strong>Figure 2:</strong> Impact of selected models' number.</em></p>
    </td>
    <td>
        <img src="./assets/figure3.png" alt="ablation" width="420">
        <p><em><strong>Figure 3:</strong> Impact of cluster count K</em></p>
    </td>
</tr>
</table>
</div>

**Robust to design choices**: Ablation studies demonstrate that The Avengers is insensitive to the choice of embedding models, clustering methods (14<K<140), and model selection. With 22 models tested, our automatic model selection based on cluster ranking (scoring) achieves peak performance with ~10 models, while using only 3 models matches GPT-4.1's average performance. Among different ensemble strategies, *Self-Consistency* shows the best performance.

## Getting Started

### Prerequisites

- Python 3.12+
- Required dependencies (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/ZhangYiqun018/Avengers.git
cd Avengers

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Usage instructions will be added after the paper is published
```

## Evaluation Benchmarks

Our method has been comprehensively evaluated across multiple categories:

| **Category** | **Benchmarks** |
|------------|----------------|
| Mathematics | AIME, Math500, LiveMathBench, *MathBench* |
| Code | MBPP, HumanEval, *StudentEval* |
| Logic | KORBench, Knights and Knaves, BBH, *Winogrande* |
| Knowledge | MMLUPro, GPQA, FinQA, MedQA, ARCC, *BrainTeaser* |
| Affective | EmoryNLP, MELD, *DailyDialog* |

*Note: Datasets in italics are used as out-of-distribution datasets in our paper.*


## Dataset

**Access to evaluation datasets**: [Google Drive](https://drive.google.com/drive/folders/1OkSxruoHSPvBCDWrMsKK1r81kK2CJrbv?usp=drive_link)

## Citation

If you find our work useful for your research, please consider citing:

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

## Contact

For questions or collaborations, please contact: [yiqunzhang@stumail.neu.edu.cn](mailto:yiqunzhang@stumail.neu.edu.cn) or [li.hao@mail.nwpu.edu.cn](mailto:li.hao@mail.nwpu.edu.cn)