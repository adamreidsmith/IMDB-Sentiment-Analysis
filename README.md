# Sentiment Analysis of IMDB Movie Ratings
### By Adam Smith

## Overview

This project aims to perform sentiment analysis on IMDB movie ratings. The goal is to classify movie reviews as either positive or negative based on their text content. Three different models are implemented and evaluated: a Naive Bayesian Network, a Convolutional Neural Network (CNN) with pretrained GloVe embeddings, and a Fully-Connected Neural Network using Keras' built-in IMDB dataset processing.

## Dependencies and Data Setup

### Dependencies
The project requires the following Python libraries:
- `re`
- `glob`
- `random`
- `itertools`
- `collections`
- `numpy`
- `matplotlib`
- `multiprocess`
- `nltk`
- `scikit-learn`
- `tensorflow`

If running in Google Colab, the notebook includes a cell to download necessary libraries and data files. This cell installs `multiprocess` and downloads NLTK resources (`stopwords`, `punkt`, `wordnet`, `omw-1.4`).

### Data

The project uses two main datasets:

1.  **ACL IMDB Dataset:** For the Naive Bayesian and Convolutional models. This dataset consists of 25,000 movie reviews for training and 25,000 for testing. The notebook downloads and extracts this dataset.
2.  **GloVe Pretrained Word Embeddings:** Used by the Convolutional model.
3.  **Keras IMDB Dataset:** For the Fully-Connected model. This dataset is loaded directly via `tensorflow.keras.datasets.imdb`.

## Data Processing

A `Dataset` class is defined to load and preprocess the ACL IMDB review data.
The preprocessing steps include:

  - Reading training and testing data from text files.
  - Removing HTML tags.
  - Removing punctuation and converting text to lowercase.
  - Tokenizing reviews.
  - Removing stopwords.
  - Lemmatizing words using their part of speech.
  - Optionally reducing the test set to only contain words present in the training data.
  - Mapping ratings to binary labels (0 for negative, 1 for positive).
  - Shuffling the training data.

Multiprocessing is used to speed up the text preprocessing.

For the Fully-Connected model, the Keras IMDB dataset is loaded and vectorized such that each review is represented as a binary vector of length `VOCAB_SIZE` (10,000), where an entry is 1 if the corresponding word is present in the review.

## Models

### 1\. Naive Bayesian Network

  - **Vectorization:** Uses `sklearn.feature_extraction.text.CountVectorizer` to create a sparse matrix representation of the reviews, where each entry is the count of a word in a review.
  - **Training:** A `sklearn.naive_bayes.MultinomialNB` model is trained on the vectorized training data.
  - **Performance:**
      - Train Accuracy: 90.27%
      - Test Accuracy: 81.98%
      - Train Set MCC: 0.806
      - Test Set MCC: 0.644

### 2\. Convolutional Neural Network (CNN)

  - **Hyperparameters:**
      - `VOCAB_SIZE`: Number of unique words in the training data.
      - `SEQUENCE_LENGTH`: 300
      - `EMBEDDING_DIM`: 100 (using GloVe pre-trained embeddings)
      - `BATCH_SIZE`: 256
      - `EPOCHS`: 5
      - `LR`: 1e-3
  - **Architecture:**
    1.  `TextVectorization` layer: Maps strings to integer sequences.
    2.  `Embedding` layer: Uses pretrained GloVe embeddings (non-trainable).
    3.  `Conv1D` (128 filters, kernel size 7, ReLU activation)
    4.  `MaxPool1D` (pool size 5)
    5.  `Conv1D` (128 filters, kernel size 5, ReLU activation)
    6.  `MaxPool1D` (pool size 5)
    7.  `Conv1D` (128 filters, kernel size 5, ReLU activation)
    8.  `GlobalMaxPooling1D`
    9.  `Dense` (32 units, ReLU activation)
    10. `Dropout` (0.5)
    11. `Dense` (1 unit, for output)
  - **Compilation:** Optimized using Adam optimizer with binary cross-entropy loss.
  - **Performance:**
      - True Training Accuracy: 94.26%
      - Validation Accuracy (Test Set): 85.15% (after 5 epochs)
      - Train Set MCC: 0.886
      - Test Set MCC: 0.705

### 3\. Fully-Connected Neural Network

  - **Hyperparameters:**
      - `VOCAB_SIZE`: 10,000
      - `LR`: 2e-4
      - `EPOCHS`: 3
      - `BATCH_SIZE`: 256
      - `L2_COEFF`: 0.002 (for L2 regularization)
  - **Data Vectorization:** Each review is converted into a 10,000-dimensional vector where each dimension indicates the presence or absence of a word.
  - **Architecture:**
    1.  `Dense` (32 units, ReLU activation, L2 kernel regularizer)
    2.  `Dropout` (0.5)
    3.  `Dense` (16 units, ReLU activation, L2 kernel regularizer)
    4.  `Dropout` (0.5)
    5.  `Dense` (1 unit, sigmoid activation for binary classification)
  - **Compilation:** Optimized using Adam optimizer with binary cross-entropy loss.
  - **Performance:**
      - Training Accuracy (reported by Keras during fit): 81.90% (Epoch 3)
      - Validation Accuracy (Test Set): 87.69% (Epoch 3)
      - True Training Accuracy (calculated after fit): 90.21%
      - Test Accuracy (calculated after fit): 87.69%
      - Train Set MCC: 0.805
      - Test Set MCC: 0.754

## Testing and Results (Fully-Connected Model)

  - **Training Accuracy:** 90.21%
  - **Testing Accuracy:** 87.69%
  - **Train Set MCC Score:** 0.805
  - **Test Set MCC Score:** 0.754

Confusion matrices for the training and test data show that the model mistakenly classifies negative reviews roughly as often as positive reviews.

## Conclusion

The project successfully implements and evaluates three different models for sentiment analysis on IMDB movie reviews. The Naive Bayesian model provides a decent baseline. The Convolutional Neural Network with pretrained embeddings shows improved performance but struggles to exceed \~85% test accuracy without overfitting. The Fully-Connected model, using Keras' built-in dataset and vectorization, achieves the target test accuracy of over 87% with measures to control overfitting (Dropout and L2 regularization). The MCC scores provide a more nuanced view of the classification performance for all models.