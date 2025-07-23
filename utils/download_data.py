import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate
api = KaggleApi()
api.authenticate()

# Download dataset
dataset_name = "mdrifaturrahman33/levir-cd"
save_dir = "./data"  # You can change this if needed
api.dataset_download_files(dataset_name, path=save_dir, unzip=True)

print("Downloaded to:", os.path.abspath(save_dir))

