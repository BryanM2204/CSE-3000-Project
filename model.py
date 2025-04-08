import pandas as pd
from data_cleanup import main, cleanup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def user_test(model_toxic, model_target, vectorizer, target_labels):
    while True:
        user_input = input("Enter a comment (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        cleaned_input = cleanup(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])

        # Predict toxicity
        pred_toxic = model_toxic.predict(input_tfidf)[0]
        toxic_label = "Toxic" if pred_toxic == 1 else "Non-Toxic"
        pred_toxic_proba = model_toxic.predict_proba(input_tfidf)

        # Predict target demographic
        pred_target = model_target.predict(input_tfidf)[0]
        pred_target_label = target_labels[pred_target]

        print(f"Prediction: {toxic_label}")
        print(f"Confidence: [Non-Toxic: {pred_toxic_proba[0][0]:.4f}, Toxic: {pred_toxic_proba[0][1]:.4f}]")
        print(f"Potentially targeting: {pred_target_label}")
        print()

# Load the cleaned data
data = main()

X = [row[1] for row in data]  # cleaned text
y_toxic = [row[0] for row in data]  # 0 or 1
y_target_raw = [row[2] for row in data]  # e.g., "gender", "ethnicity"

# Convert target categories to numbers
target_categories = list(set(y_target_raw))
target_category_to_num = {cat: i for i, cat in enumerate(target_categories)}
y_target = [target_category_to_num[target] for target in y_target_raw]

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train-Test split
X_train, X_test, y_toxic_train, y_toxic_test, y_target_train, y_target_test = train_test_split(
    X_tfidf, y_toxic, y_target, test_size=0.2, random_state=42
)

# Train the Toxicity model
model_toxic = LogisticRegression(max_iter=1000)
model_toxic.fit(X_train, y_toxic_train)

# Train the Target Demographic model
model_target = LogisticRegression(max_iter=1000)
model_target.fit(X_train, y_target_train)

# Evaluate Toxicity
y_toxic_pred = model_toxic.predict(X_test)
print("Toxicity Classification Report:\n", classification_report(y_toxic_test, y_toxic_pred))
print("Toxicity Confusion Matrix:\n", confusion_matrix(y_toxic_test, y_toxic_pred))

# Evaluate Target
y_target_pred = model_target.predict(X_test)
print("\nTarget Demographic Classification Report:\n", classification_report(y_target_test, y_target_pred))
print("Target Demographic Confusion Matrix:\n", confusion_matrix(y_target_test, y_target_pred))

# User Testing
user_test(model_toxic, model_target, vectorizer, target_categories)
