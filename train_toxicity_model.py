from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from bert_dataset import load_and_prepare_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report
)
import seaborn as sns
import torch

# Load dataset
dataset = load_and_prepare_dataset(
    "Dynamically Generated Hate Dataset v0.2.3.csv",
    binary_bias=True
)

# Load model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    ignore_mismatched_sizes=True  # Add this line
)

# Custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return {
        'accuracy': accuracy,
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'f1': 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / (tp / (tp + fp) + tp / (tp + fn)),
        'tnr': tn / (tn + fp)
    }

# Training arguments with evaluation
# Change 'evaluation_strategy' to 'eval_strategy'
training_args = TrainingArguments(
    output_dir="./results_toxicity",
    eval_strategy="epoch",  # Correct parameter name
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

# Train and save model
trainer.train()
model.save_pretrained("./results_toxicity")

# Generate final evaluation metrics and plots
def plot_metrics(test_dataset):
    # Get predictions
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(
        torch.Tensor(predictions.predictions), 
        dim=-1  # Corrected parameter
    )
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./results_toxicity/confusion_matrix.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, probs[:, 1], name='Toxicity Model')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.savefig('./results_toxicity/roc_curve.png')
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_true, probs[:, 1], name='Toxicity Model')
    plt.title('Precision-Recall Curve')
    plt.savefig('./results_toxicity/precision_recall_curve.png')
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=['Non-Toxic', 'Toxic'])
    with open('./results_toxicity/classification_report.txt', 'w') as f:
        f.write(report)

# Generate visualizations
plot_metrics(dataset['test'])
print("Visualizations saved to ./results_toxicity/ directory")