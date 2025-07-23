# File: run_train.py

from train import train

if __name__ == "__main__":
    train(
        data_dir='data/LEVIR_CD',  # adjust this path
        epochs=10,
        batch_size=4,
        lr=1e-3
    )
