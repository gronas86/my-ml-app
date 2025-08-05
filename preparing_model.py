import pandas as pd
from typing import Dict
from datasets import Dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import Trainer, TrainingArguments
from preparing_data import prepare_full_data, prepare_sample_of_data




def from_pd_to_dataset(data)-> Dict[str, pd.DataFrame]:
    """
    Convert a pandas DataFrame to a Hugging Face Dataset.

    Args:
        data {str: pd.DataFrame}: The DataFrame to convert.

    Returns:
        {str: datasets.Dataset}: The converted Dataset.
        Just the train and eval datasets.
    """
    if not isinstance(data, dict):
        raise ValueError("Input data must be a Dict.")
    # Create a dictionary of datasets from the input data
    datasets_dict = data.copy()
    datasets_dict.pop('test', None)
    # Convert each DataFrame in the dictionary to a Hugging Face Dataset
    for key, value in datasets_dict.items():
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame.")
        datasets_dict[key] = Dataset.from_pandas(value)
    return datasets_dict

def tokenize_dataset(data) -> Dict[str, object]:
    """Tokenize the datasets using DistilBERT tokenizer.
    Returns: Dict of tokenized datasets and the tokenizer."""
    dataset_dict = from_pd_to_dataset(data)
    # Create a DistilBERT tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased")

    # Define a function to tokenize the dataset
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length", # Add padding to the sequences
            truncation=True       # Truncate sequences longer than the
        )                         # model's max length

    # Tokenize the train and eval datasets
    tokenized_train = dataset_dict['train'].map(
        tokenize_function, batched=True)
    tokenized_eval = dataset_dict['eval'].map(
        tokenize_function, batched=True)

    return {
        "train": tokenized_train,
        "eval": tokenized_eval,
        "tokenizer": tokenizer
    }

def config_model(
        tokenized_data,
        epochs=3,
        batch_size=8,
        wights_decay=0.01,
        learning_rate=2e-5) -> Dict[str, object]:
    """Prepare and config the DistilBERT model and
    pipeline for text classification.
    Returns: Dict of objects including the model, trainer, and pipeline."""
    # Create a DistilBERT model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    )
    # Define training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=wights_decay,
        logging_steps=10
    )
    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['eval']
    )
    # Create a text classification pipeline
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenized_data['tokenizer'],
        device=-1  # Use CPU for inference, change to 0 for GPU
    )
    return {
        "training_args": training_args,
        "trainer": trainer,
        "pipeline": pipe
    }

if __name__ == "__main__":
    data = prepare_sample_of_data(10)
    tokenized_data = tokenize_dataset(data)
    model_data = config_model(tokenized_data)
    print(model_data)