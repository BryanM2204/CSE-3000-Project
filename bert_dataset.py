import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

def load_and_prepare_dataset(file_path, binary_bias=False):
    # Load and preprocess data
    df = pd.read_csv(file_path)
    df = df[['text', 'label', 'target']]

    if binary_bias:
        # Binary classification setup (toxicity/bias detection)
        df['labels'] = df['target'].apply(
            lambda t: 0 if t == 'none' else 1
        ).astype(int)
        df = df[['text', 'labels']]  # Keep only necessary columns
        
    else:
        # Multiclass setup (target group classification)
        df = df[(df['target'] != 'none') & (df['target'] != 'notgiven')]
        df['label'] = df['label'].apply(
            lambda x: 1 if x == 'hate' else 0
        ).astype(int)
        
        # Create target mapping
        target_labels = df['target'].unique().tolist()
        target2id = {label: i for i, label in enumerate(target_labels)}
        id2target = {i: label for label, i in target2id.items()}
        
        df['labels'] = df['target'].map(target2id).astype(int)
        df = df[['text', 'labels']]

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(['text'])  # Remove original text column
    
    # Train-test split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    if binary_bias:
        return dataset
    else:
        return dataset, target_labels, target2id, id2target