import pandas as pd
from data_cleanup import main, cleanup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from visualization import plot_confusion_matrix
# from linear_regression import run_linear_regression
# import numpy as np


def user_test(model, vectorizer):
    while True:
        user_input = input("Enter a tweet (or type 'exit' to quit): ")
            
        if user_input.lower() == 'exit':
            break
        
        cleaned_input = cleanup(user_input)
        
        input_tfidf = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_tfidf)
        prediction_proba = model.predict_proba(input_tfidf)
        label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
        
        print(f"Prediction: {label}")
        print(f"Confidence: [Non-Toxic: {prediction_proba[0][0]:.4f}, Toxic: {prediction_proba[0][1]:.4f}]\n")

data = main()

X = [row[1] for row in data]  
y = [row[0] for row in data] 

# Convert labels to binary (0 for Non-Toxic, 1 for Toxic)
y = [1 if label == 'Toxic' else 0 for label in y]

vectorizer = TfidfVectorizer(max_features=5000)  
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

user_test(model, vectorizer)

plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix')

# np.random.seed(42)
# X_lr = 2 * np.random.rand(100, 1)  # Feature
# y_lr = 4 + 3 * X_lr + np.random.randn(100, 1)  # Target
# run_linear_regression(X_lr, y_lr)