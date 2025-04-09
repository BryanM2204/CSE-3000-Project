from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from bert_dataset import load_and_prepare_dataset

# Load dataset WITH removing "none"
dataset, target_labels, target2id, id2target = load_and_prepare_dataset(
    "Dynamically Generated Hate Dataset v0.2.3.csv",
    remove_none_targets=True  # Remove "none" examples for Bias model
)

# Rename "target_id" column to "labels"
dataset['train'] = dataset['train'].rename_column("target_id", "labels")
dataset['test'] = dataset['test'].rename_column("target_id", "labels")

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(target_labels))

# Training settings
training_args = TrainingArguments(
    output_dir="./results_target",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

trainer.train()
model.save_pretrained("./results_target")
