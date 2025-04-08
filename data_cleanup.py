import csv
import re
import string
import nltk
from nltk.corpus import stopwords

# Make sure you have stopwords downloaded
nltk.download('stopwords')

def cleanup(tweet):
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#\w+", "", tweet)
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    stop_words = set(stopwords.words("english"))
    tweet = " ".join([word for word in tweet.split() if word.lower() not in stop_words])
    tweet = tweet.lower()
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

def tokenize_csv(file_path):
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            label = row['label'].strip().lower()
            target = row['target'].strip().lower()
            text = row['text']

            # Skip invalid rows
            if not text or label not in ['hate', 'nothate']:
                continue

            # Convert labels to Toxic / Non-Toxic
            toxicity = 1 if label == 'hate' else 0

            # Cleanup the text
            cleaned_text = cleanup(text)

            tokens.append((toxicity, cleaned_text, target))

    return tokens

def main():
    file_path = "../Dynamically Generated Hate Dataset v0.2.3.csv"  # <-- Update your path if needed
    tokens = tokenize_csv(file_path)
    return tokens

if __name__ == "__main__":
    print(main())
