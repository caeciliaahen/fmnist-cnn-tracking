# File: train.py

import yaml
from src.train import train

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train(
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"]
    )