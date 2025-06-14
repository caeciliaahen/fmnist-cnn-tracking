# File: train.py

import argparse
import yaml
from src.train import train

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=params.get("epochs", 10))
    parser.add_argument("--batch_size", type=int, default=params.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=params.get("lr", 0.001))

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )