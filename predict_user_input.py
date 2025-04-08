from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer and models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_toxic = BertForSequenceClassification.from_pretrained("./results_toxicity")
model_target = BertForSequenceClassification.from_pretrained("./results_target")

# Mapping
target_labels = [...]  # <-- paste your list of target labels here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_toxic.to(device)
model_target.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Toxicity
    outputs_toxic = model_toxic(**inputs)
    pred_toxic = torch.argmax(outputs_toxic.logits, dim=1).item()

    # Target
    outputs_target = model_target(**inputs)
    pred_target = torch.argmax(outputs_target.logits, dim=1).item()

    toxic_label = "Toxic" if pred_toxic == 1 else "Non-Toxic"
    target_label = target_labels[pred_target]

    print(f"Prediction: {toxic_label}")
    print(f"Target demographic (if toxic): {target_label}")

if __name__ == "__main__":
    while True:
        text = input("Enter a comment (or type 'exit'): ")
        if text.lower() == 'exit':
            break
        predict(text)
