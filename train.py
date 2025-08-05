import os
import shutil
import time
import mlflow
from sklearn.metrics import accuracy_score


def get_accuracy_metrics(test_data, pipe):
    """Get accuracy metrics for the test data using the provided pipeline.
    Uses sklearn's accuracy_score to compute the accuracy."""
    test_texts = test_data["text"].tolist()
    test_labels = test_data["label"].tolist()
    predicted_labels = [
        int(pred["label"].split("_")[-1]) for pred in pipe(test_texts)
    ]
    return accuracy_score(test_labels, predicted_labels)

def config_mlflow(experiment_name='ag_news_distilbert'):
    """Configure MLflow tracking URI and experiment."""
    dir_path = './mlruns'
    os.makedirs(dir_path, exist_ok=True)
    mlflow.set_tracking_uri(dir_path)
    mlflow.set_experiment(experiment_name)
    mlflow.transformers.autolog(log_datasets=True)
    run_name = f"{experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    return run_name


def save_model_locally(pipe, run_name):
    model_save_path = f"./models/{run_name}"
    if os.path.exists(model_save_path):
        if os.listdir(model_save_path):
            shutil.rmtree(model_save_path)
            mlflow.transformers.save_model(
                transformers_model=pipe,
                task="text-classification",
                path=model_save_path
            )
    else:
        mlflow.transformers.save_model(
            transformers_model=pipe,
            task="text-classification",
            path=model_save_path
        )

def train_model(run_name, test_data, pipe, trainer, training_args):
    """Train the model using MLflow and log metrics and artifacts."""
    with mlflow.start_run(run_name=run_name):
        # Log model parameters and training configuration
        mlflow.log_params({
            "model_name": "distilbert-base-uncased",
            "num_labels": 4,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "train_batch": training_args.per_device_train_batch_size,
            "eval_batch": training_args.per_device_eval_batch_size,
        })

        # Model training
        trainer.train()

        # Log steps quantity
        mlflow.log_metric("train_total_steps", trainer.state.max_steps)
        mlflow.log_metric(
            "steps_per_epoch",
            trainer.state.max_steps / training_args.num_train_epochs
        )
        # Log accuracy metrics in test data
        accuracy = get_accuracy_metrics(test_data, pipe)
        mlflow.log_metric("test_accuracy", accuracy)

        # Evaluate the model and log evaluation metrics
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: float(v) for k, v in eval_metrics.items()})

        # Log the model DistilBERT
        mlflow.transformers.log_model(
            transformers_model=pipe,
            task="text-classification",
            artifact_path="model",
            registered_model_name='ag_news_distilbert'
        )

        # Save the model to a local path
        while True:
            ask = input('Do you want to save the model locally? (y/n) ' 
                  f'\nModel will be saved to ./models/{run_name}): ')
            if ask.lower().strip() == 'y':
                save_model_locally(pipe, run_name)
                print('Model saved locally.')
                return
            elif ask.lower().strip() == 'n':
                print('Model not saved locally.')
                return
            else:
                print('Please enter "y" or "n".')

if __name__ == '__main__':
    from preparing_data import prepare_sample_of_data
    from preparing_model import from_pd_to_dataset, tokenize_dataset, config_model
    # Prepare sample data
    data = prepare_sample_of_data(10)
    # Convert DataFrame to Dataset
    dataset_dict = from_pd_to_dataset(data)
    # Tokenize the dataset
    tokenized_data = tokenize_dataset(dataset_dict)
    # Configure the model
    model_config = config_model(
        tokenized_data,
        epochs=3,
        batch_size=8,
        wights_decay=0.01,
        learning_rate=2e-5
    )
    # Configure MLflow
    run_name = config_mlflow()
    # Train the model and log metrics
    train_model(
        run_name,
        data['test'],
        model_config['pipeline'],
        model_config['trainer'],
        model_config['training_args'])