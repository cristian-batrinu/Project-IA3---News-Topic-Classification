# News Topic Classification: Adversarial Attacks and Defense Mechanisms

This project implements adversarial attacks (HotFlip, TextFooler) and defense mechanisms (Adversarial Training, Defensive Distillation) on BERT and TextCNN models for news topic classification using the AG News dataset.

## Overview

The massive volume of news data requires efficient automated classification systems. While deep learning models like BERT and TextCNN achieve high accuracy, they are vulnerable to adversarial attacks that subtly modify text to induce classification errors. This project explores these vulnerabilities and evaluates defense mechanisms.

## What's Included

- **Models:**
  - BERT pre-trained model (`lucasresck/bert-base-cased-ag-news`) [[12]](#references)
  - TextCNN (Convolutional Neural Network for text classification)
  
- **Attacks:**
  - HotFlip attack - white-box gradient-based attack [[5]](#references)
  - TextFooler attack - black-box word replacement attack [[7]](#references)
  
- **Defenses:**
  - Adversarial training defense [[9]](#references), [[10]](#references)
  - Defensive distillation defense [[6]](#references), [[11]](#references)
  
- **Features:**
  - Pre-computed BERT embeddings for faster training
  - Comprehensive metrics tracking (accuracy, recall, F1, precision, attack success rate)
  - Parallelized data loading and attack generation

## Dataset

**AG News Dataset** - A popular dataset for text classification applications [[1]](#references), [[2]](#references), [[3]](#references)

- **Training Set:** 108,000 examples (5% subset used: ~5,400 samples)
- **Test Set:** 7,600 examples (5% subset used: ~380 samples)
- **Labels:** World, Sports, Business, Science (4 classes)
- **Features:** News article text

**Note:** This project uses a 5% subset of the dataset for computational efficiency. Literature typically uses full datasets or smaller subsets (0.2-1%) for faster experiments. See `DATASET_SUBSET_NOTE.md` for details.

## Algorithms

### HotFlip Attack

Based on Ebrahimi et al. (2018) [[5]](#references). HotFlip is a white-box attack method that modifies characters to generate adversarial examples using gradient-based optimization. The attack uses three operations:
- **Character flip** - fastest to generate
- **Character insertion**
- **Character deletion**

The best change is calculated based on gradients, selecting the word vector that presents the largest increase in loss. The attack typically requires fewer than 5 token modifications.

**Mathematical Formulation:**
For a sequence x, a character flip at position j in word i can be represented as a one-hot vector. The maximum loss is calculated as:

$$\max \nabla_x J(\mathbf{x}, \mathbf{y})^T \cdot \vec{v}_{ijb} = \max_{ijb} \frac{\partial J}{\partial x_{ij}}^{(b)} - \frac{\partial J}{\partial x_{ij}}^{(a)}$$

Files: `src/train_test_hotflip.py`, `src/train_test_textcnn_hotflip.py`

### TextFooler Attack

Based on Jin et al. (2020) [[7]](#references). TextFooler is a black-box attack that maintains semantic meaning while fooling the model. The algorithm:

1. Ranks words based on their impact on sentence meaning
2. Filters stop words
3. Searches for candidate synonyms for each important word
4. Filters candidates based on semantic similarity
5. Checks semantic similarity of each candidate with the original word
6. Iteratively replaces words until the attack succeeds

**Parameters:** Maximum word perturbations = 10

Files: `src/train_test_textfooler.py`, `src/train_test_textfooler_original.py`, `src/train_test_textcnn_textfooler.py`

### Adversarial Training

Trains the model with a mix of clean and adversarial examples to improve robustness [[9]](#references), [[10]](#references). For BERT, 70% of each batch consists of adversarial examples. Separate models are trained for each attack type.

**Hyperparameters:**
- BERT: 3 epochs, batch size 16, learning rate 2e-5, adversarial fraction 0.7
- TextCNN: 20 epochs, batch size 2048, learning rate 1e-3, filter windows 3, 4, or 5

Files: `src/train_test_adversarial_training.py`, `src/train_test_textcnn_adversarial_training.py`

### Defensive Distillation

Uses a teacher model (baseline) to train a student model with soft labels [[6]](#references), [[11]](#references). The teacher model produces "soft" probabilities using a temperature (5 for BERT, 20 for TextCNN), generating a lower confidence score for the majority class. The student model learns to imitate the teacher's mechanism for detecting erroneous instances.

**Hyperparameters:**
- BERT: Temperature = 5.0, Alpha = 0.5, 3 epochs, batch size 16, learning rate 2e-5
- TextCNN: Temperature = 20, Alpha = 0.5, 20 epochs, batch size 2048, learning rate 1e-3

Files: `src/train_test_defensive_distillation.py`, `src/train_test_textcnn_defensive_distillation.py`

## Installation

You'll need:
- Python 3.8 or higher
- PyTorch
- CUDA (optional but recommended)

Clone the repo:
```bash
git clone https://github.com/Edy14Borta/Proiect_IA3.git
cd Proiect_IA3
```

Create virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

Install packages:
```bash
pip install torch transformers scikit-learn numpy pandas tqdm nltk datasets
```

NLTK data downloads automatically when you run the scripts. Or download manually:
```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

The model `lucasresck/bert-base-cased-ag-news` downloads automatically on first run.

## How to Run

Run scripts in this order:

### 1. Preprocess and cache embeddings

```bash
python src/preprocess_and_cache.py
```

Only run this once. It loads the dataset, splits it, preprocesses texts (lowercase, tokenize, remove stopwords, stem), extracts BERT embeddings, and saves them.

### 2. Train baseline

```bash
python src/train_test_baseline.py
```

Trains a baseline model and saves it to `models/baseline_model/`.

### 3. Train with attacks

```bash
python src/train_test_textfooler.py
python src/train_test_hotflip.py
```

These train models and test them under attack.

### 4. Train with defenses

```bash
python src/train_test_adversarial_training.py hotflip
python src/train_test_adversarial_training.py textfooler
python src/train_test_defensive_distillation.py hotflip
python src/train_test_defensive_distillation.py textfooler
```

Note: defensive distillation needs the baseline model from step 2.

## Project Structure

```
Proiect/
├── src/
│   ├── preprocess_and_cache.py          # Dataset preprocessing and embedding caching
│   ├── train_test_baseline.py           # BERT baseline training
│   ├── train_test_textfooler.py         #TextFooler attack on BERT
│   ├── train_test_hotflip.py            # HotFlip attack on BERT
│   ├── testare_hotflip.py    # HotFlip attack on TextCNN
│   ├── testare_textfooler.py # TextFooler attack on TextCNN
│   ├── train_test_adversarial_training.py      # Adversarial training for BERT
│   ├── testare_adversarial_training.py # Adversarial training for TextCNN
│   ├── train_test_defensive_distillation.py    # Defensive distillation for BERT
│   ├── testare_defensive_distillation.py # Defensive distillation for TextCNN
│   └── utils.py                          # Utility functions for parallelization
├── models/                    # Saved model checkpoints
├── cached_embeddings/         # Pre-computed BERT embeddings
├── test_results/              # CSV results files
├── README.md
└── bibliography.md
```

## Configuration

### BERT Model Settings
- MODEL_NAME: `'lucasresck/bert-base-cased-ag-news'` [[12]](#references)
- SUBSET_PERCENTAGE: `0.05` (5% of dataset, ~5,400 training samples)
- MAX_LENGTH: `128`
- BATCH_SIZE: `16` (baseline, attacks, defenses)
- NUM_EPOCHS: `3`
- LEARNING_RATE: `2e-5` [[12]](#references)

### TextCNN Model Settings
- Epochs: `20` (baseline), `10-20` (attacks/defenses)
- Batch Size: `512-2048`
- Learning Rate: `1e-3`
- Filter Windows: `3, 4, 5`
- Number of Filters: `100`
- Embedding Dimension: `768` (uses BERT embeddings)

### Attack Parameters
- **HotFlip:** max_perturbations = 5 [[5]](#references)
- **TextFooler:** max_perturbations = 10, similarity_threshold = 0.7, synonym_num = 50 [[7]](#references)

### Defense Parameters
- **Adversarial Training:** adversarial_fraction = 0.7 (70% of batch) [[10]](#references)
- **Defensive Distillation:** temperature = 5.0 (BERT), 20.0 (TextCNN), alpha = 0.5 [[6]](#references), [[11]](#references)

## Results

Each script saves:
1. Console output with comprehensive metrics
2. Model checkpoint in `models/`
3. CSV file in `test_results/` with all metrics and hyperparameters

CSV files include train/val/test accuracy, recall, F1, precision, attack accuracy, attack success rate (ASR), and all hyperparameters used.

View results:
```bash
ls test_results/
cat test_results/baseline_results_*.csv
```

## Experimental Results

### Baseline Performance

**BERT Pre-trained Model:**
- Test Accuracy: ~85-90% (on 5% subset)
- Precision (Weighted): ~85-90%
- F1 Score (Weighted): ~85-90%

**TextCNN:**
- Test Accuracy: ~84-91% (varies with hyperparameters)
- Precision (Weighted): ~85-91%
- F1 Score (Weighted): ~85-91%

### Attack Results

#### HotFlip Attack

**BERT:**
- Attack Success Rate (ASR): ~19%
- Accuracy after attack: ~85%
- Precision (Weighted): ~84.84%
- F1 Score (Weighted): ~85.24%

**TextCNN:**
- Attack Success Rate (ASR): ~44.4%
- Accuracy after attack: ~59.86%
- Precision (Weighted): ~61.62%
- F1 Score (Weighted): ~60.20%

#### TextFooler Attack

**BERT:**
- Attack Success Rate (ASR): ~20%
- Accuracy after attack: ~86.32%
- Precision (Weighted): ~86.21%
- F1 Score (Weighted): ~86.72%

**TextCNN:**
- Attack Success Rate (ASR): ~20.31%
- Accuracy after attack: ~84.96%
- Precision (Weighted): ~85.06%
- F1 Score (Weighted): ~84.98%

### Defense Results

#### Adversarial Training

**BERT:**
- **Against HotFlip:** Accuracy ~87.37%, Precision ~87.24%, F1 ~87.51%
- **Against TextFooler:** Accuracy ~86.58%, Precision ~86.54%, F1 ~87.01%

**TextCNN:**
- **Against HotFlip:** Accuracy ~91.29%, Precision ~91.33%, F1 ~91.27%
- **Against TextFooler:** Accuracy ~51.79%, Precision ~53.29%, F1 ~49.87%

#### Defensive Distillation

**BERT:**
- **Against HotFlip:** Accuracy ~86.32%, Precision ~86.27%, F1 ~86.75%
- **Against TextFooler:** Accuracy ~85.53%, Precision ~85.48%, F1 ~85.73%

**TextCNN:**
- **Against HotFlip:** Accuracy ~66.78%, Precision ~67.05%, F1 ~66.89%
- **Against TextFooler:** Accuracy ~53.79%, Precision ~53.69%, F1 ~53.72%

## Key Findings

1. **HotFlip vs TextFooler:** HotFlip, being gradient-based, can sometimes choose replacements that maximize mathematical error but are less semantically natural than TextFooler's cosine similarity-filtered synonyms.

2. **Model Robustness:** BERT shows better robustness than TextCNN under adversarial attacks, maintaining higher accuracy after attacks.

3. **Adversarial Training:** Effective for improving robustness, especially for BERT models. Shows "overfitting" to HotFlip attack type for TextCNN.

4. **Defensive Distillation:** Shows minimal effectiveness for text classification, consistent with literature findings [[6]](#references). May not provide significant robustness gains.

5. **Dataset Size:** Using at least a larger percentage of the dataset (beyond 5%) may be necessary to observe notable differences in attack metrics for BERT models.

6. **CNN Vulnerability:** TextCNNs are more vulnerable to small perturbations and harder to recover after attacks compared to BERT.





## Background

News topic classification has been extensively studied in recent years [[1]](#references), [[2]](#references), [[3]](#references). The problem involves categorizing news articles into predefined topics (World, Sports, Business, Science) to enable automated content filtering, news recommendation, and misinformation detection.

### Adversarial Attacks

Deep learning models like BERT and TextCNN achieve high accuracy but are vulnerable to adversarial attacks:

- **HotFlip** demonstrates that minimal character-level changes can significantly degrade model performance [[5]](#references)
- **TextFooler** shows that semantically-preserving word replacements can fool models [[7]](#references)

These attacks reveal that models can drop 30-50 percentage points in accuracy under adversarial conditions.

### Defense Mechanisms

Defense strategies aim to improve model robustness:

- **Adversarial Training** trains models on a mix of clean and adversarial examples [[9]](#references), [[10]](#references). Can reduce attack success rates and improve robustness by 20-40 percentage points.
- **Defensive Distillation** uses soft probability distributions from a teacher model to train a more robust student model [[6]](#references), [[11]](#references). However, literature suggests minimal effectiveness for text classification tasks.

## Future Directions

- Testing attacks generated on CNN against Transformer models (BERT) to simulate realistic Black-box scenarios
- Identifying token sequences that can compromise the model regardless of input
- Implementing FreeLB to reduce the computational cost of adversarial training on Transformer architectures
- Using Attention Maps to visualize how adversarial perturbations alter the model's semantic focus
- Exploring other machine learning models and attack/defense techniques for potentially higher performance

## References

[1] P. Sunagar, A. Kanavalli, S. S. Nayak, S. R. Mahan, S. Prasad, S. Prasad, "News Topic Classification Using Machine Learning Techniques", in V. Bindhu, J.M.R.S. Tavares, AA. A. Boulogeorgos, C. Vuppalapati (Eds.) *International Conference on Communication, Computing and Electronics Systems. Lecture Notes in Electrical Engineering,* vol. 733, Springer, Singapore, pp. 461-474, Mar. 2021

[2] L. Serreli, C. Marche and M. Nitti, "Reducing Data Volume in News Topic Classification: Deep Learning Framework and Dataset," *IEEE Open Journal of the Computer Society*, vol. 6, pp. 153-164, Jan. 2025

[3] S. Daud, M. Ullah, A. Rehman, T. Saba, R. Damasevicius, A. Sattar, "Topic Classification of Online News Articles Using Optimized Machine Learning Models", *Computers,* vol. 12, iss. 1, Jan. 2023

[4] J. Wong, "Exploring TEXTFOOLER's Syntactically- and Semantically-Sound Distributed Adversarial Attack Methodology", *CS224n: Natural Language Processing with Deep Learning,* 2020

[5] J. Ebrahimi, A. Rao, D. Lowd, D. Dou, "HotFlip: White-Box Adversarial Examples for Text Classification", in I. Gurievich, Y. Mikao (Eds.) *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),* vol. 2, Melbourne, Australia, pp. 31-36, Jul. 2018

[6] M. Soll, T. Hinz, S. Magg, S. Wermter, "Evaluating Defensive Distillation for Defending Text Processing Neural Networks Against Adversarial Examples", in I. Tetko, V. Kůrková, P. Karpov, F. Theis (Eds.) *Artificial Neural Networks and Machine Learning – ICANN 2019: Image Processing. ICANN 2019. Lecture Notes in Computer Science,* vol. 11729, Springer, Cham, pp. 685-696, Sep. 2019

[7] D. Jin, Z. Jin, J. T. Zhou, P. Szolovits, "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment", in *Proceedings of the AAAI Conference on Artificial Intelligence,* vol. 34, no. 05, pp. 8018-8025, Apr. 2020

[8] J. Morris, E. Lifland, J. Y. Yoo, J. Grigsby, D. Jin, Y. Qi, "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP", in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations,* pp. 119-126, Nov. 2020

[9] A. Shafahi, M. Najibi, M. A. Ghiasi, Z. Xu, J. Dickerson, C. Studer, L. S. Davis, G. Taylor, T. Goldstein, "Adversarial Training for Free!", in *Advances in Neural Information Processing Systems,* vol. 32, Dec. 2019

[10] F. Nikfam, M. Rezaei, M. H. Mozaffari, M. Shafique, "AccelAT: A Framework for Accelerating the Adversarial Training of Deep Neural Networks through Accuracy Gradient", *arXiv preprint arXiv:2210.06888,* Oct. 2022

[11] G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network", in *NIPS 2015 Deep Learning Workshop,* Dec. 2015

[12] J. Devlin, M. Chang, K. Lee, K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", in *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,* vol. 1, pp. 4171-4186, Jun. 2019

## Additional Resources

- `bibliography.md` - Complete bibliography with all references
- Model: https://huggingface.co/lucasresck/bert-base-cased-ag-news
- Dataset: Hugging Face datasets library (AG News)
- TextAttack Framework: https://github.com/QData/TextAttack [[8]](#references)

## Implementation Details

### Libraries Used
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - BERT models and tokenizers
- **NLTK** - Natural language processing (WordNet, POS tagging, stopwords)
- **scikit-learn** - Machine learning utilities and metrics
- **datasets** - Dataset loading and management

### Performance Optimizations
- Pre-computed BERT embeddings for faster training
- Parallelized data loading with multiple workers
- GPU acceleration support (CUDA)
- Batch processing for attack generation

## License

Educational/research use only.

## Acknowledgments

- Hugging Face for transformers library and model hosting
- Authors of the research papers cited
- AG News dataset creators
- NLTK project
- TextAttack framework developers [[8]](#references)

## Contact

For issues or questions: https://github.com/cristian-batrinu/Project-IA3---News-Topic-Classification
