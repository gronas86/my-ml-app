from data_loader import load_main_test_data_from_s3
from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split



def prepare_full_data() -> Dict[str, pd.DataFrame]:
    """
    Prepare the big data by loading it from S3 and splitting it into
    train and eval sets.

    Returns:
       {"train": train_df, "eval": eval_df, "test": test_df}:
       A dictionary containing the train, eval, test DataFrames.
    """
    # Load the main and test data from S3
    main_df, test_df = load_main_test_data_from_s3()
    if main_df is None or test_df is None:
        raise ValueError("Failed to load data from S3.")

    # Split the main data into train and eval sets
    train_df, eval_df = train_test_split(
        main_df, test_size=0.2, random_state=42)

    return {"train": train_df, "eval": eval_df, "test": test_df}


def prepare_sample_of_data(snippet_size=100) -> Dict[str, pd.DataFrame]:
    """ Prepare a small sample of the data for quick testing.
        Returns:
        {
        "train": small_train_data,
        "eval": small_eval_data,
        "test": small_test_data
        }:
        A dictionary containing the small train, eval, test DataFrames.
    """
    # Load the full data from S3 and split it into train, eval, test sets
    dataset_dict = prepare_full_data()
    train_df = dataset_dict["train"]
    eval_df = dataset_dict["eval"]
    test_df = dataset_dict["test"]

    # Sample a small portion of the data
    small_train_data = train_df.sample(n=snippet_size, random_state=42)
    small_eval_data = eval_df.sample(n=snippet_size, random_state=42)
    small_test_data = test_df.sample(n=snippet_size, random_state=42)
    print(f"Snippets size: {len(small_train_data)}")

    return {
        "train": small_train_data,
        "eval": small_eval_data,
        "test": small_test_data
    }

if __name__ == "__main__":
    # Example usage
    full_data = prepare_full_data()
    sample_data = prepare_sample_of_data(snippet_size=10)


