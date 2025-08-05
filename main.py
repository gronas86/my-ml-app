import sys
import questionary
from preparing_data import prepare_full_data, prepare_sample_of_data
from preparing_model import tokenize_dataset, config_model
from train import config_mlflow, train_model

if __name__ == "__main__":

# Initialize the program with a choice of datasets size
    mode_1 = questionary.select(
        "Choose datasets size:",
        choices=["Full", "Sample", "Exit"]
    ).ask()
# If full data is selected, we prepare the full dataset
    if mode_1 == "Full":
        data = prepare_full_data()
        print("Full data prepared successfully.")

# If sample data is selected, we prepare a sample of the dataset
    elif mode_1 == "Sample":
        while True:
            size_input = questionary.text(
                "Enter the size of the sample (default is 100):",
                default="100"
            ).ask()
            try:
                if size_input.lower() == 'q':
                     sys.exit(0)
                size = int(size_input)

                if size <= 0:
                    raise ValueError("Sample size must be positive.")
                break

            except ValueError:
                print("Invalid input. Please enter a positive integer.")
        data = prepare_sample_of_data(int(size))
# If Exit or else we exit the program
    else:
        print("Exiting the program.")
        sys.exit(0)

# After the data is prepared, we proceed with tokenization
    print("\nInitializing tokenization and model configuration...")
    try:
        tokenized_data = tokenize_dataset(data)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        sys.exit(1)
    else:
        print("Tokenization completed successfully.\n")

# After tokenization we proceed with model configuration
    mode_2 = questionary.select(
        "Choose the model configuration:",
        choices=["Default", "Custom", "Exit"]
    ).ask()

# Default configuration uses the default parameters for DistilBERT
    if mode_2 == "Default":

        model_config = config_model(tokenized_data)
        print("Model configured with default settings.")

# Custom configuration allows the user to specify epochs, batch size, and weight decay
    elif mode_2 == "Custom":
# Setting epochs with error handling
        while True:
            epochs = questionary.text(
                "Enter the number of epochs (default is 3):",
                default="3"
            ).ask()
            try:
                if epochs.lower() == 'q':
                     sys.exit(0)
                epochs_int = int(epochs)

                if epochs_int <= 0:
                    raise ValueError("Epochs size must be positive.")
                break

            except ValueError:
                print("Invalid input. Please enter a positive integer.")
# Setting batch size with error handling
        while True:
            batch_size = questionary.text(
                "Enter the batch size (default is 8):",
                default="8"
            ).ask()
            try:
                if batch_size.lower() == 'q':
                     sys.exit(0)
                b_size = int(batch_size)

                if b_size <= 0:
                    raise ValueError("Batch size must be positive.")
                break

            except ValueError:
                print("Invalid input. Please enter a positive integer.")
# Setting weight decay with error handling
        while True:
            weight_decay = questionary.text(
                "Enter the weight decay (default is 0.01):",
                default="0.01"
            ).ask()
            try:
                if weight_decay.lower() == 'q':
                     sys.exit(0)

                weights = float(weight_decay)

                if weights < 0:
                    raise ValueError("Weight decay must be positive.")
                elif not (0 <= weights <= 0.1):
                    raise ValueError("Weight decay must be between 0 and 0.1")
                break

            except ValueError as e:
                print(f"Invalid input: {e}.")

# If all inputs are valid, we proceed with model configuration

        try:

            model_config = config_model(
                tokenized_data=tokenized_data,
                epochs=int(epochs),
                batch_size=int(batch_size),
                wights_decay=float(weight_decay)
            )
            print("Model configured with custom settings.")
        except ValueError as e:
            print(f"Error in model configuration: {e}")
# If Exit we exit the program
    else:
        print("Exiting the program.")
        sys.exit(0)

# After the model is configured, we proceed with MLflow configuration
    mode_3 = questionary.select(
        "Choose MLflow configuration:",
        choices=["Default", "Custom", "Exit"]
    ).ask()

# If Default configuration is selected, we use the default experiment name
    if mode_3 == "Default":
        run_name = config_mlflow()
        print(f"MLflow configured with default settings. Run name: {run_name}")

# If Custom we set a custom experiment name
    elif mode_3 == "Custom":
        experiment_name = questionary.text(
            "Enter the experiment name (default is 'ag_news_distilbert')",
            default="ag_news_distilbert"
        ).ask()
        run_name = config_mlflow(experiment_name)
        print(f"MLflow configured with custom settings. Run name: {run_name}")

# If Exit we exit the program
    else:
        print("Exiting the program.")
        sys.exit(0)

# Finally, we proceed with model training
    train_model(
        run_name,
        data['test'],
        model_config['pipeline'],
        model_config['trainer'],
        model_config['training_args'])

