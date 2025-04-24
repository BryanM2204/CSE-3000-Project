from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from bert_dataset import load_and_prepare_dataset

dataset = load_and_prepare_dataset("Dynamically Generated Hate Dataset v0.2.3.csv", binary_bias=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training settings
training_args = TrainingArguments(
    output_dir="./results_bias_binary",
    evaluation_strategy="epoch",
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
model.save_pretrained("./results_bias_binary")
