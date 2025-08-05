import os
from dotenv import load_dotenv
import boto3
import pandas as pd
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

def load_main_test_data_from_s3():
    try:
        # Load environment variables
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
        TRAIN_PATH = os.getenv("S3_TRAIN_PATH")
        TEST_PATH = os.getenv("S3_TEST_PATH")

        assert all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME,
            TRAIN_PATH, TEST_PATH]),"Some environment variables are missing!"

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )

        # Load path and bucket name from S3
        bucket_name = BUCKET_NAME
        train_path = TRAIN_PATH
        test_path = TEST_PATH

        # Get the train and test objects from S3
        train_obj = s3_client.get_object(Bucket=bucket_name, Key=train_path)
        test_obj = s3_client.get_object(Bucket=bucket_name, Key=test_path)

        # Read the CSV files from the S3 objects
        main_df = pd.read_csv(BytesIO(train_obj['Body'].read()))
        test_df = pd.read_csv(BytesIO(test_obj['Body'].read()))

        print("S3 data loaded successfully")
        return main_df, test_df

    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    main_df, test_df = load_main_test_data_from_s3()
