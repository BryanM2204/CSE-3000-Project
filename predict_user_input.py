from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer and models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_toxic = BertForSequenceClassification.from_pretrained("./results_toxicity", local_files_only=True)
model_bias = BertForSequenceClassification.from_pretrained("./results_bias_binary", local_files_only=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_toxic.to(device)
model_bias.to(device)

# Rule-based bias keyword detection
BIAS_KEYWORDS = {
    "gender": ["woman", "man", "girl", "boy", "feminist", "trans"],
    "race": ["black", "white", "asian", "mexican", "arab", "hispanic"],
    "religion": ["muslim", "jew", "christian", "hindu", "islamic"],
    "disability": ["retard", "cripple", "dumb", "disabled", "r3t4rd"],
    "nationality": ["american", "chinese", "african", "european"]
}

def detect_bias_keywords(text):
    text = text.lower()
    matches = set()
    for category, keywords in BIAS_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matches.add(category)
    return list(matches)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Toxicity prediction
    outputs_toxic = model_toxic(**inputs)
    pred_toxic = torch.argmax(outputs_toxic.logits, dim=1).item()
    toxic_conf = torch.softmax(outputs_toxic.logits, dim=1)[0][pred_toxic].item()

    # Bias prediction (binary)
    outputs_bias = model_bias(**inputs)
    pred_bias = torch.argmax(outputs_bias.logits, dim=1).item()
    bias_conf = torch.softmax(outputs_bias.logits, dim=1)[0][pred_bias].item()

    matched_keywords = detect_bias_keywords(text)

    toxic_label = "Toxic" if pred_toxic == 1 else "Non-Toxic"
    bias_label = "Targets a demographic group" if pred_bias == 1 else "Not group-targeted"

    print(f"\nText: {text}")
    print(f"Prediction \u2192 {toxic_label} (Confidence: {toxic_conf:.2%})")
    print(f"Bias       \u2192 {bias_label} (Confidence: {bias_conf:.2%})")
    if matched_keywords:
        print(f"Bias types \u2192 {', '.join(matched_keywords)}")

if __name__ == "__main__":
    while True:
        text = input("Enter a comment (or type 'exit'): ")
        if text.lower() == 'exit':
            break
        predict(text)
