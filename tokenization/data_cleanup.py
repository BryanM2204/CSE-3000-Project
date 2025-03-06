import csv
import re
import string
import nltk
from nltk.corpus import stopwords


def tokenize_csv(file_path):
    """
    Tokenizes a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of lists, where each inner list represents a row of tokens.
              Returns an empty list if an error occurs during file processing.
    """
    try:
        tokens = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                try:
                    col1 = float(row[1])  
                    col4 = float(row[4])  
                except (ValueError, IndexError):
                    print(f"Skipping row due to invalid data: {row}")
                    continue

                # Perform the comparison and determine the replacement word
                if (col1 - col4) > col4:
                    replacement = ["Toxic"]  
                else:
                    replacement = ["Non-Toxic"]  

                tweet = row[6]
                cleaned_tweet = cleanup(tweet)

                modified_row = replacement + [cleaned_tweet]
                tokens.append(modified_row)

        return tokens
    
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

def cleanup(tweet):
    tweet = re.sub(r"@\w+", "", tweet)
    
    # Remove hashtags (#example)
    tweet = re.sub(r"#\w+", "", tweet)
    
    # Remove punctuation and special characters
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tweet = " ".join([word for word in tweet.split() if word.lower() not in stop_words])
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove extra spaces
    tweet = re.sub(r"\s+", " ", tweet).strip()
    
    return tweet

def main():
    file_path = "../labeled_data.csv"
    tokens = tokenize_csv(file_path)


    return tokens


print(main())