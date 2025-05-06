# Sentiment Analysis of Customer Reviews
# Using TF-IDF Vectorization and Logistic Regression

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
import time
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("Libraries imported successfully!")

# 1. Data Preparation
# For demonstration purposes, we'll use a sample dataset
# In a real-world scenario, you might load your own CSV file

# Sample data creation (for demonstration)
# You can replace this with your own data loading code
def create_sample_data(n_samples=1000):
    """Create a sample dataset of customer reviews with sentiment labels"""
    
    positive_texts = [
        "Absolutely loved this product! Best purchase I've made all year.",
        "Great customer service and quick delivery.",
        "The quality exceeded my expectations. Would definitely recommend!",
        "Amazing value for money, will purchase again.",
        "Fantastic product, works exactly as described.",
        "Very satisfied with my purchase, excellent quality.",
        "Outstanding service and product quality.",
        "Incredible value, exceeded all my expectations!",
        "This product has made my life so much easier!",
        "Superb quality and excellent customer service.",
    ]
    
    negative_texts = [
        "Terrible product, broke after one use.",
        "Very disappointed with the quality, would not recommend.",
        "Customer service was unhelpful and rude.",
        "Product arrived damaged and return process was difficult.",
        "Waste of money, does not work as advertised.",
        "Poor quality and overpriced. Avoid this product.",
        "Extremely disappointed, product doesn't work at all.",
        "Awful experience from start to finish.",
        "The worst purchase I've made in years.",
        "Complete waste of money, avoid at all costs.",
    ]
    
    neutral_texts = [
        "Product is okay, nothing special.",
        "Does the job but could be better.",
        "Average quality for the price.",
        "Not bad, but not great either.",
        "Decent product but shipping took too long.",
        "It works as expected, no complaints.",
        "Received as described, nothing more nothing less.",
        "It's alright, just what I needed.",
        "Satisfactory product, meets basic requirements.",
        "Neither impressed nor disappointed with this purchase.",
    ]
    
    # Generate random reviews based on the templates above
    np.random.seed(42)
    samples = []
    
    for _ in range(n_samples):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.4, 0.2])
        
        if sentiment == 'positive':
            text = np.random.choice(positive_texts)
            rating = np.random.choice([4, 5])
        elif sentiment == 'negative':
            text = np.random.choice(negative_texts)
            rating = np.random.choice([1, 2])
        else:
            text = np.random.choice(neutral_texts)
            rating = 3
            
        # Add some random noise to the text to make it more diverse
        words = text.split()
        if len(words) > 5 and np.random.random() > 0.5:
            # Randomly remove some words
            indices_to_keep = np.random.choice(len(words), size=int(len(words) * 0.8), replace=False)
            words = [words[i] for i in sorted(indices_to_keep)]
            text = ' '.join(words)
            
        samples.append({
            'review_text': text,
            'rating': rating,
            'sentiment': sentiment
        })
    
    return pd.DataFrame(samples)

# Create or load dataset
df = create_sample_data(1000)

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Number of reviews: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
df.head()

# Check class distribution
print("\nSentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# Visualize sentiment distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Display rating distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='rating', data=df)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 2. Text Preprocessing
def preprocess_text(text):
    """
    Preprocess text data by performing:
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Removing stop words
    - Lemmatization
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Apply preprocessing to the review text
print("\nApplying text preprocessing...")
start_time = time.time()
df['processed_text'] = df['review_text'].apply(preprocess_text)
print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

# Display original and processed text for comparison
print("\nOriginal vs Processed Text Examples:")
comparison_df = df[['review_text', 'processed_text']].head()
print(comparison_df)

# 3. Feature Engineering with TF-IDF
# For binary sentiment classification, convert to binary labels
print("\nConverting to binary sentiment classification...")
df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 1})
print(f"Binary sentiment distribution:\n{df['sentiment_binary'].value_counts()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], 
    df['sentiment_binary'], 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=5,         # Minimum document frequency
    max_df=0.8,       # Maximum document frequency
    max_features=5000 # Limit features to prevent overfitting
)

# Transform training data
print("\nTransforming text data with TF-IDF vectorization...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")

# Display top features (words with highest TF-IDF scores)
def display_top_features(vectorizer, top_n=10):
    """Display top features based on their IDF scores"""
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_
    
    # Sort features by IDF scores
    feature_scores = sorted(zip(feature_names, idf_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} features by IDF score:")
    for feature, score in feature_scores[:top_n]:
        print(f"{feature}: {score:.4f}")

display_top_features(tfidf_vectorizer)

# 4. Model Training: Logistic Regression
print("\nTraining Logistic Regression model...")
start_time = time.time()

# Create and train the model
log_reg = LogisticRegression(
    C=1.0,               # Regularization strength (inverse)
    max_iter=1000,       # Maximum iterations for convergence
    random_state=42,
    solver='liblinear',  # Efficient solver for small datasets
    class_weight='balanced' # Handle class imbalance
)

log_reg.fit(X_train_tfidf, y_train)
print(f"Model training completed in {time.time() - start_time:.2f} seconds")

# 5. Model Evaluation
# Make predictions
y_pred = log_reg.predict(X_test_tfidf)
y_pred_prob = log_reg.predict_proba(X_test_tfidf)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 6. Feature Importance Analysis
# Extract and visualize the most important features for sentiment classification
def plot_important_features(model, vectorizer, n_top_features=20):
    """Plot the most important features for classification"""
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients from the model
    coef = model.coef_[0]
    
    # Create DataFrame for visualization
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    })
    
    # Sort by absolute coefficient value (importance)
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(n_top_features)
    
    # Plot positive and negative coefficients
    plt.figure(figsize=(12, 8))
    
    # Sort for visualization
    coef_df = coef_df.sort_values('Coefficient')
    
    # Color code based on sentiment direction
    colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
    
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(f'Top {n_top_features} Important Features for Sentiment Classification')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Positive Sentiment'),
        Line2D([0], [0], color='red', lw=4, label='Negative Sentiment')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

# Visualize the most important features
plot_important_features(log_reg, tfidf_vectorizer)

# 7. Hyperparameter Tuning with Grid Search
print("\nPerforming hyperparameter tuning with Grid Search...")

# Create a pipeline to streamline the process
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(random_state=42))
])

# Parameters to tune
param_grid = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__min_df': [2, 5],
    'tfidf__max_df': [0.7, 0.8],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__class_weight': ['balanced', None]
}

# Use a smaller parameter grid for demonstration purposes
# In a real project, you might want to expand this grid
small_param_grid = {
    'tfidf__max_features': [5000],
    'tfidf__min_df': [5],
    'classifier__C': [0.1, 1.0]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    small_param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all available cores
)

# For the notebook demonstration, we'll use a subset of data to make it run faster
# In a real project, you would use the full dataset
X_sample = X_train[:500]  # Using a subset for demonstration
y_sample = y_train[:500]

grid_search.fit(X_sample, y_sample)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# 8. Final Model with Best Parameters
print("\nTraining final model with best parameters...")

# Get best parameters
best_params = grid_search.best_params_

# Create and train final model with best parameters
final_tfidf = TfidfVectorizer(
    max_features=best_params['tfidf__max_features'],
    min_df=best_params['tfidf__min_df']
)

X_train_final = final_tfidf.fit_transform(X_train)
X_test_final = final_tfidf.transform(X_test)

final_model = LogisticRegression(
    C=best_params['classifier__C'],
    random_state=42,
    max_iter=1000
)

final_model.fit(X_train_final, y_train)

# Make predictions with the final model
y_pred_final = final_model.predict(X_test_final)
y_pred_prob_final = final_model.predict_proba(X_test_final)[:, 1]

# Evaluate final model
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")

print("\nFinal Model Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Negative', 'Positive']))

# Plot feature importance for final model
plot_important_features(final_model, final_tfidf, n_top_features=15)

# 9. Practical Application: Sentiment Prediction Function
def predict_sentiment(text, vectorizer, model):
    """
    Predict sentiment of new text using the trained model
    Returns: sentiment (0=negative, 1=positive) and probability
    """
    # Preprocess the text
    processed = preprocess_text(text)
    
    # Transform using the vectorizer
    text_tfidf = vectorizer.transform([processed])
    
    # Make prediction
    sentiment = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0][sentiment]
    
    return sentiment, probability

# Test the prediction function with some examples
print("\nTesting sentiment prediction on new reviews:")

test_reviews = [
    "This product is amazing! I would definitely recommend it to everyone.",
    "Terrible experience, would not buy again. Complete waste of money.",
    "It's okay, not the best but does the job adequately.",
    "Shipping was fast but the product quality is questionable."
]

for review in test_reviews:
    sentiment, prob = predict_sentiment(review, final_tfidf, final_model)
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"\nReview: '{review}'")
    print(f"Predicted sentiment: {sentiment_label} with {prob:.2%} confidence")

# 10. Conclusion and Summary
print("""
# Sentiment Analysis Project Summary

## Key Findings:
1. Successfully built a sentiment analysis model using TF-IDF vectorization and Logistic Regression
2. Achieved good classification performance on customer reviews
3. Identified the most important words/features for sentiment prediction
4. Created a reusable pipeline for sentiment analysis of new reviews

## Future Improvements:
1. Collect more diverse real-world data
2. Experiment with advanced models (e.g., SVM, BERT, Transformers)
3. Implement more sophisticated text preprocessing techniques
4. Consider multi-class sentiment classification (e.g., very negative, negative, neutral, positive, very positive)
5. Deploy the model as a web service for real-time sentiment analysis

## Applications:
- Customer feedback analysis
- Social media monitoring
- Product review summarization
- Brand sentiment tracking
- Customer support prioritization
""")

print("\nSentiment Analysis project completed successfully!")