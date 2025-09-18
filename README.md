# NLP: Disaster Tweet Classification

> **Note:** The notebook in this repository is a clean version with all cell outputs cleared for fast rendering on GitHub.
>
> **➡️ [View the fully rendered notebook on NBViewer](https://nbviewer.org/github/shahrosek/disaster-tweets-nlp-classifier/blob/main/disaster-tweet-classification-v1.ipynb)**
>
> **➡️ [View the original notebook on Kaggle](https://www.kaggle.com/code/shahrosek/disaster-tweet-classification-v1)**

This repository contains the solution for the Kaggle competition "Natural Language Processing with Disaster Tweets." The primary goal is to build a model that can accurately determine whether a given tweet is about a real disaster or not.

## 🎯 Objective
To classify tweets into two categories: disaster-related (1) or not disaster-related (0). This involves significant text preprocessing and the application of a robust classification model.

## 📊 Dataset
The project uses the official dataset from the [Kaggle "Disaster Tweets" competition](https://www.kaggle.com/c/nlp-getting-started), which includes tweet text, keywords, and location data.

## ⚙️ Methodology & Technical Walkthrough

1.  **Data Cleaning & Exploration**:
    * Analyzed the dataset for missing values using the `missingno` library.
    * Dropped uninformative columns (`location`) and cleaned the `keyword` feature by removing URL encoding.

2.  **Advanced Text Preprocessing**:
    * Developed a comprehensive pipeline to clean and normalize the tweet text, including:
        * Unicode normalization and removal of user mentions and URLs.
        * Expansion of contractions (e.g., "don't" -> "do not").
        * Conversion of emojis to their text descriptions (e.g., 🔥 -> "fire").
        * **Hashtag Segmentation**: Splitting concatenated hashtags (e.g., `#SanFranciscoFire`) into constituent words ("San Francisco Fire").
        * Removal of punctuation, numbers, and common English stopwords using `NLTK`.
        * Application of **Porter Stemming** to reduce words to their root form.

3.  **Feature Engineering & Modeling**:
    * Combined the cleaned `text` and `keyword` fields into a single input feature.
    * Converted the text corpus into a numerical format using `CountVectorizer` with a **bi-gram** range `(1, 2)` to capture two-word phrases, which adds valuable context.
    * Trained and evaluated three models: Gaussian Naive Bayes, Random Forest, and **CatBoost**.

## 📈 Results & Outcome
The **CatBoost Classifier**, trained on bi-gram features, proved to be the most effective model.
* **Accuracy:** **80.3%**
* **F1-Score:** **0.74**

The final notebook produces a `submission.csv` file with predictions for the test set.

## 🛠️ Tech Stack
* **Language**: `Python`
* **Libraries**: `Pandas`, `NumPy`, `NLTK`, `Scikit-learn`, `CatBoost`, `emoji`, `contractions`, `wordsegment`, `missingno`.
