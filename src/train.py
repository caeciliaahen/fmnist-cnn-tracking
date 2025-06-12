# File: src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import mlflow
import os
import json
import csv

from src.model import build_model
from src.data import get_dataloaders
from src.evaluate import evaluate
from datetime import datetime

def train(epochs=10, batch_size=64, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)

    tb_writer = SummaryWriter(log_dir="runs")
    mlflow.set_experiment("fashion_cnn_experiment")
    os.makedirs("outputs", exist_ok=True)

    metrics_csv_path = "outputs/metrics.csv"
    with open(metrics_csv_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch)

            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            tb_writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            with open(metrics_csv_path, mode="a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)  # Jangan pakai tb_writer!
                csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_acc])

            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        model_path = "outputs/model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device, save_dir="outputs")
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

        metrics = {"test_loss": test_loss, "test_acc": test_acc}
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("outputs/metrics.json")
        mlflow.log_artifact("outputs/confusion_matrix.csv")
        mlflow.log_artifact("outputs/predictions.csv")
        mlflow.log_artifact("outputs/classification_report.json")

        tb_writer.add_scalar("Loss/Test", test_loss, epochs)
        tb_writer.add_scalar("Accuracy/Test", test_acc * 100, epochs)

        tb_writer.close()