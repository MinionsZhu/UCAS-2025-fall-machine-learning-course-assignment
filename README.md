# UCAS-2025-fall-machine-learning-course-assignment
This is a part of a course assignment UCAS 2025 fall machine learning.

## Background[^1]
Based on various phrases extracted from movie reviews (these phrases come from real movie reviews on the Rotten Tomatoes website), predict a sentiment label for each phrase using a 5-level scale:

- **negative** (negative / very negative)
- **somewhat negative** (somewhat negative)
- **neutral** (neutral)
- **somewhat positive** (somewhat positive)
- **positive** (positive / very positive)

Find detailed question description, including data format description, through this [Kaggle link](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews, "sentiment analysis on movie reviews").
## Method
We have 3 models for this task, each based on **BoW** (Bag-of-Words), **SVM** (Support Vector Mechine) and Transformer respectively. This repo implements the model based on **BoW** and offers some figures that comes from training and evaluating process.

Because the basic **BoW** (Bag-of-Words) method is too simple and has many limitations, we adopted an improved approach proposed in a paper published in 2023[^2].

Briefly, in this article the author used BoW and MLP for classification:
- **Text Representation**: TF-IDF vectors using the top 30,000 unigrams, L2-normalized.
- **Classifier**: Multilayer Perceptron (MLP) with the following architecture:
  - Input: 30,000-dimensional TF-IDF vector
  - Hidden Layer 1: 500 units, ReLU activation
  - Hidden Layer 2: 500 units, ReLU activation
  - Output: Softmax (single-label) or Sigmoid (multi-label)
- **Regularization**: Dropout and early stopping
- **Loss**: Cross-entropy (single-label) or Binary cross-entropy (multi-label)

This simple yet wide MLP on TF-IDF consistently outperforms many complex graph-based models.
In our implementation, we added a smaller hidden layer after the existing layers, with half units of them to improve the model's nonlinear ability.
## Dataset and Data Processing
### Dataset Description
The dataset consists of tab-separated files(.tsv format) derived from the Rotten Tomatoes movie reviews. Sentences are parsed into multiple phrases using the Stanford parser, with the original train/test split preserved (though sentences are shuffled). Duplicate phrases (e.g., common short words) appear only once.

- **train.tsv**: Contains PhraseId, SentenceId, Phrase, and Sentiment label.
- **test.tsv**: Contains PhraseId, SentenceId, and Phrase (no labels; to be predicted).

**Sentiment Labels**:
- 0 — negative
- 1 — somewhat negative
- 2 — neutral
- 3 — somewhat positive
- 4 — positive
### Data Processing
Since we chose to use BoW as the base method, the data processing process is quite simple:

- clean dataset: Retain letters, numbers, and basic punctuation; remove excess whitespace. Do not remove the words "not", "very", or "but".
- Split the validation set using **group-based cross-validation**
  - Use the `SentenceId` column from `train.tsv` as the **group ID**.
  - Reason: In `train.tsv`, different phrases extracted from the same sentence are treated as separate samples. To avoid data leakage, all phrases from the same sentence must stay entirely within either the training set or the validation set.
- **TF-IDF Vectorization (BoW)**: 
  - Incorporates bigrams to capture semantic reversals (e.g., "not good").
  - Vocabulary limited to the top 30,000 most frequent terms.
## Training Proccess

## Bagging and Stacking

## Results

## References
[^1]: Will Cukierski. (2014). *Sentiment Analysis on Movie Reviews*. Kaggle. Available at: https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews
[^2]: Galke, L. (2023). *Are We Really Making Much Progress? Bag-of-Words vs. Sequence vs. Graph vs. Hierarchy for Single-label and Multi-label Text Classification*. https://api.semanticscholar.org/CorpusID:268090830
