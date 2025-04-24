import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import defaultdict

# Load your fine-tuned toxicity model
model = BertForSequenceClassification.from_pretrained("./results_toxicity", local_files_only=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Demographic categories and keywords
DEMOGRAPHIC_KEYWORDS = {
    "gender": ["woman", "man", "girl", "boy", "feminist", "trans"],
    "race": ["black", "white", "asian", "mexican", "arab", "hispanic"],
    "religion": ["muslim", "jew", "christian", "hindu", "islamic"],
    "nationality": ["american", "chinese", "african", "european"]
}

# Neutral sentence templates
TEMPLATES = [
    "I support {} people.",
    "{} people are strong.",
    "I work with many {} individuals.",
    "I believe in the rights of {} people.",
    "I admire many {} people."
]

# Predict toxicity
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred  # 1 = toxic, 0 = not toxic

# Run the audit
results = defaultdict(lambda: {"total": 0, "toxic": 0})

for group, keywords in DEMOGRAPHIC_KEYWORDS.items():
    for keyword in keywords:
        for template in TEMPLATES:
            sentence = template.format(keyword)
            prediction = predict_toxicity(sentence)
            results[group]["total"] += 1
            if prediction == 1:
                results[group]["toxic"] += 1

# Report results
print("\nToxicity Bias Audit Results:")
print("----------------------------------")
for group, data in results.items():
    total = data["total"]
    toxic = data["toxic"]
    rate = toxic / total if total > 0 else 0
    print(f"{group.capitalize():<12} → Toxic rate: {rate:.2%} ({toxic}/{total})")
