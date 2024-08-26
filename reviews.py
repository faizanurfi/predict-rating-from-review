import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat.textstat import textstat
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

root_dir = r'/Users/faizanurfi/Desktop/BU Grad/Summer 24/CS677/Final Project/furfi_final_project/'
file = os.path.join(root_dir, 'AllProductReviews.csv')
review_body_col = 'ReviewBody'
review_star_col = 'ReviewStar'


def cleanup_preprocess(df):
    # Drop duplicates
    df = df.drop_duplicates(subset=review_body_col, keep='first')

    # Remove empty
    df = df.dropna(subset=[review_body_col])
    df = df[df[review_body_col].str.strip().astype(bool)]

    # Remove stop words
    df[review_body_col] = df[review_body_col].apply(remove_stop_words)

    # Fix stem text
    df[review_body_col] = df[review_body_col].apply(stem_text)


def remove_stop_words(text):
    stop_words = stopwords.words('english')
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])


def stem_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words])


def len_analysis(len_col):
    # Checking length quality
    bins = np.arange(0, 200, 5)
    length_counts, bin_edges = np.histogram(len_col, bins=bins)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], length_counts, width=np.diff(bin_edges), edgecolor='black', align='edge', color='skyblue')

    # Customize plot
    plt.xlabel('Review Length Range (words)')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Review Lengths')
    plt.xticks(bin_edges, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show plot
    plt.savefig(os.path.join(root_dir, 'Review_Length_Distribution.png'))


def sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']


def extract_features(row):
    review_body = row[review_body_col]
    polarity = TextBlob(review_body).sentiment.polarity
    subjectivity = TextBlob(review_body).sentiment.subjectivity
    sentiment = sentiment_score(review_body)
    body_len = len(review_body.split())
    reading_ease = textstat.flesch_reading_ease(review_body)
    reading_grade = textstat.flesch_kincaid_grade(review_body)
    return pd.Series([polarity, subjectivity, sentiment, body_len, reading_ease, reading_grade])


def predict_and_evaluate(classifier, name, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f'Prediction Analysis for: {name}')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


try:
    print('--------------------------------------------------')
    print('Data Cleaning and Preprocessing')
    print('--------------------------------------------------')
    df = pd.read_csv(file)

    # Cleaning and preprocessing
    cleanup_preprocess(df)

    print('--------------------------------------------------')
    print('Feature Engineering')
    print('--------------------------------------------------')

    features = ['polarity', 'subjectivity', 'sentiment', 'body_len', 'reading_ease', 'reading_grade']

    df[features] = df.apply(extract_features, axis=1)

    print('--------------------------------------------------')
    print('Length Analysis')
    print('--------------------------------------------------')

    # Length Analysis
    len_analysis(df[features[3]])

    print('--------------------------------------------------')
    print('Pairwise Correlation Analysis')
    print('--------------------------------------------------')

    correlation_matrix = df[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Pairwise Correlation Heatmap')
    plt.savefig(os.path.join(root_dir, 'Correlation_Heatmap.png'))

    print('--------------------------------------------------')
    print('Scatterplot - Sentiment vs Polarity')
    print('--------------------------------------------------')

    # Scatter plot of Sentiment vs. polarity
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=features[2], y=features[0], data=df, hue='ReviewStar', palette='viridis')
    plt.title('Sentiment vs. Polarity')
    plt.xlabel('Sentiment')
    plt.ylabel('Polarity')
    plt.legend(title='Review Star')
    plt.savefig(os.path.join(root_dir, 'Polarity_Sentiment_Heatmap.png'))

    print('--------------------------------------------------')
    print('Model Training, Test & Evaluation - 5 Star Rating System')
    print('--------------------------------------------------')

    x = df[features]
    y = df[review_star_col]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    log_reg = LogisticRegression()
    predict_and_evaluate(log_reg, 'Logistic Regression', X_train, X_test, y_train, y_test)

    gradient_boosting = GradientBoostingClassifier()
    predict_and_evaluate(gradient_boosting, 'Gradient Boosting', X_train, X_test, y_train, y_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    predict_and_evaluate(knn, 'KNN', X_train, X_test, y_train, y_test)

    svm = SVC()
    predict_and_evaluate(svm, 'SVM', X_train, X_test, y_train, y_test)

    random_forest = RandomForestClassifier()
    predict_and_evaluate(random_forest, 'Random Forest', X_train, X_test, y_train, y_test)

    decison_tree = DecisionTreeClassifier()
    predict_and_evaluate(decison_tree, 'Decision Tree', X_train, X_test, y_train, y_test)

    classifiers = {
        "Logistic Regression": log_reg,
        "SVM": svm,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "KNN": knn,
        "Decision Tree": decison_tree
    }

    results = {}
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

    df_metrics = pd.DataFrame(results).T
    df_metrics.reset_index(inplace=True)
    df_metrics.rename(columns={'index': 'Classifier'}, inplace=True)

    df_melted = df_metrics.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Classifier', y='Score', hue='Metric', data=df_melted, palette='viridis')

    plt.title('Model Performance Metrics')
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'Performance_Metrics_5_star_system.png'))

    print('--------------------------------------------------')
    print('Model Training, Test & Evaluation - Binary Rating System')
    print('--------------------------------------------------')

    review_label_col = 'ReviewLabel'
    binary_rating = df[df[review_star_col] != 3]
    binary_rating[review_label_col] = binary_rating[review_star_col].apply(lambda x: 1 if x > 3 else 0)

    X = binary_rating[review_body_col]
    y = binary_rating[review_label_col]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    metrics = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)

        tp = conf_matrix[1, 1]
        fn = conf_matrix[1, 0]
        fp = conf_matrix[0, 1]
        tn = conf_matrix[0, 0]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'TPR': tpr,
            'TNR': tnr
        }

        print(metrics[model_name], accuracy)

    metrics_df = pd.DataFrame(metrics).T

    metrics_df.plot(kind='bar', figsize=(14, 8), alpha=0.7)

    plt.title('Performance Metrics for Binary Classification (0 vs 1 Reviews)')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig(os.path.join(root_dir, 'Performance_Metrics_Binary_classification.png'))


except Exception as e:
    print(e)
