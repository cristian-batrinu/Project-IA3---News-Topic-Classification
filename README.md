# Adversarial Attacks and Defenses for BERT Text Classification

This project implements adversarial attacks (HotFlip, TextFooler) and defense mechanisms (Adversarial Training, Defensive Distillation) on BERT models for text classification using the AG News dataset.

## What's Included

- HotFlip attack - white-box gradient-based attack [[5]](#references)
- TextFooler attack - black-box word replacement attack [[4]](#references)
- Adversarial training defense
- Defensive distillation defense [[6]](#references)
- Pre-computed BERT embeddings for faster training
- Metrics tracking (accuracy, recall, F1, precision)

## Algorithms

### HotFlip Attack

Based on Ebrahimi et al. (2018) [[5]](#references). HotFlip uses gradients to find tokens to flip that will mess up the model's predictions. It's pretty efficient and usually only needs to change a few tokens (<5).

File: `src/train_test_hotflip.py`

### TextFooler Attack

Based on Jin et al. (2020) via Wong (2020) [[4]](#references). TextFooler replaces words with synonyms from WordNet to fool the model while keeping the meaning similar. Works as a black-box attack.

File: `src/train_test_textfooler.py`

### Adversarial Training

Trains the model with a mix of clean and adversarial examples. By default it uses 30% adversarial examples in each batch. You train separate models for each attack type.

File: `src/train_test_adversarial_training.py`

### Defensive Distillation

Uses a teacher model to train a student model with soft labels. Temperature is set to 5.0. Combines KL divergence loss with regular cross-entropy. Needs the baseline model first.

File: `src/train_test_defensive_distillation.py`

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
│   ├── preprocess_and_cache.py
│   ├── train_test_baseline.py
│   ├── train_test_textfooler.py
│   ├── train_test_hotflip.py
│   ├── train_test_adversarial_training.py
│   └── train_test_defensive_distillation.py
├── models/                    # Saved models
├── cached_embeddings/         # Cached embeddings
├── test_results/              # CSV results
├── README.md
└── bibliography.md
```

## Configuration

All scripts use these settings:
- MODEL_NAME: `'lucasresck/bert-base-cased-ag-news'`
- SUBSET_PERCENTAGE: `0.002` (0.2% of dataset)
- MAX_LENGTH: `128`
- BATCH_SIZE: `16-32` (depends on script)
- NUM_EPOCHS: `3`
- LEARNING_RATE: `2e-5`

## Results

Each script saves:
1. Console output with metrics
2. Model checkpoint in `models/`
3. CSV file in `test_results/` with all metrics

CSV files include train/val/test accuracy, recall, F1, precision, attack accuracy, attack success rate, and hyperparameters.

View results:
```bash
ls test_results/
cat test_results/baseline_results_*.csv
```

## Expected Performance

Based on papers and my tests:

**Baseline:** Around 90-95% accuracy on AG News (depends on subset size). With 0.2% subset expect 92-94%.

**HotFlip attack:** Usually drops accuracy by 30-50 points. Attack success rate around 40-60%. Changes <5 tokens.

**TextFooler attack:** Drops accuracy by 40-60 points. Attack success rate 50-70%. Changes <10 words.

**Adversarial training:** Improves attack accuracy by 20-40 points. Reduces attack success rate by 20-40% compared to undefended model.

**Defensive distillation:** Results vary. Some papers say it doesn't help much for text classification [[6]](#references).





## Background

News classification has been studied a lot [[1]](#references). Recent work focuses on using less data [[2]](#references) and optimizing models [[3]](#references).

Adversarial attacks show BERT is vulnerable:
- HotFlip shows minimal token changes can break it [[5]](#references)
- TextFooler shows synonym replacements work [[4]](#references)

BERT can drop 80-90 percentage points under attack.

Defenses try to make models more robust:
- Adversarial training trains on adversarial examples. Can reduce attack success from 85% to 30% on AG News.
- Defensive distillation uses soft labels [[6]](#references). But might not work well for text classification.

## References

[1] P. Sunagar, A. Kanavalli, S. S. Nayak, S. R. Mahan, S. Prasad, S. Prasad, "News Topic Classification Using Machine Learning Techniques", in V. Bindhu, J.M.R.S. Tavares, AA. A. Boulogeorgos, C. Vuppalapati (Eds.) *International Conference on Communication, Computing and Electronics Systems. Lecture Notes in Electrical Engineering,* vol. 733, Springer, Singapore, pp. 461-474, Mar. 2021

[2] L. Serreli, C. Marche and M. Nitti, "Reducing Data Volume in News Topic Classification: Deep Learning Framework and Dataset," *IEEE Open Journal of the Computer Society*, vol. 6, pp. 153-164, Jan. 2025

[3] S. Daud, M. Ullah, A. Rehman, T. Saba, R. Damasevicius, A. Sattar, "Topic Classification of Online News Articles Using Optimized Machine Learning Models", *Computers,* vol. 12, iss. 1, Jan. 2023

[4] J. Wong, "Exploring TEXTFOOLER's Syntactically- and Semantically-Sound Distributed Adversarial Attack Methodology", *CS224n: Natural Language Processing with Deep Learning,* 2020

[5] J. Ebrahimi, A. Rao, D. Lowd, D. Dou, "HotFlip: White-Box Adversarial Examples for Text Classification", in I. Gurievich, Y. Mikao (Eds.) *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),* vol. 2, Melbourne, Australia, pp. 31-36, Jul. 2018

[6] M. Soll, T. Hinz, S. Magg, S. Wermter, "Evaluating Defensive Distillation for Defending Text Processing Neural Networks Against Adversarial Examples", in I. Tetko, V. Kůrková, P. Karpov, F. Theis (Eds.) Artificial Neural Networks and Machine Learning – ICANN 2019: Image Processing. ICANN 2019. Lecture Notes in Computer Science(), vol 11729. Springer, Cham, pp. 685-696, Sep. 2019

## More Info

- bibliography.md - full bibliography
- Model: https://huggingface.co/lucasresck/bert-base-cased-ag-news
- Dataset: Hugging Face datasets library

## License

Educational/research use only.

## Thanks

- Hugging Face for transformers
- Authors of the papers
- AG News dataset
- NLTK

For issues: https://github.com/cristian-batrinu/Project-IA3---News-Topic-Classification
