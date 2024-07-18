# Spam Email Filter

## Description
This project aims to develop and compare multiple machine learning models for email spam classification. By employing various text processing techniques and classification algorithms, this repository includes all the necessary code to train, evaluate, and deploy these classifiers.

## Data Preprocessing

The data preprocessing step involves cleaning the email text using various NLP techniques. This includes:

- Removing email addresses, non-alphabetic characters, and HTML tags
- Tokenizing and converting text to lowercase
- Removing stopwords
- Lemmatizing the tokens

## Feature Extraction

### Non-Neural Network-Based Embeddings

We utilize Count Vectorizer and TF-IDF Vectorizer for feature extraction from the cleaned text data.

- **Count Vectorizer**: Converts text into a matrix of token counts.
- **TF-IDF Vectorizer**: Transforms text into a matrix based on term frequency-inverse document frequency.

### Neural Network-Based Embeddings

For neural network-based embeddings, we employ Word2Vec, Doc2Vec, and BERT models.

- **Word2Vec**: Creates word embeddings using the skip-gram model.
- **Doc2Vec**: Generates document embeddings.

## Training and Evaluation

We train Logistic Regression and Random Forest classifiers on the extracted features. The models are evaluated based on accuracy, precision, recall, and F1 score. Each model is saved for future use.

## Model Evaluation Results

The table below summarizes the performance metrics of the different models trained using various feature extraction techniques:
|    | Model Name    | Embedding        |   Precision  |   Recall  |   F1_score  |   Accuracy  |
|---:|:--------------|:-----------------|-------------:|----------:|------------:|------------:|
|  0 | Logistic      | Count Vectorizer |     0.99572  |  0.930667 |    0.962095 |    0.976283 |
|  1 | Logistic      | TF_IDF           |     0.998553 |  0.92     |    0.957668 |    0.973696 |
|  2 | Logistic      | Word2Vec         |     0.990371 |  0.96     |    0.974949 |    0.984045 |
|  3 | Logistic      | Doc2Vec          |     0.955182 |  0.909333 |    0.931694 |    0.956878 |
|  4 | Decision Tree | TF_IDF           |     0.951482 |  0.941333 |    0.946381 |    0.965502 |
|  5 | Decision Tree | Count Vectorizer |     0.946667 |  0.946667 |    0.946667 |    0.965502 |
|  6 | Decision Tree | Word2Vec         |     0.961111 |  0.922667 |    0.941497 |    0.962915 |
|  7 | Decision Tree | Doc2Vec          |     0.837653 |  0.818667 |    0.828051 |    0.890039 |
