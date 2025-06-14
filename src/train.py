# File: src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import os
import json
import csv

from src.model import build_model
from src.data import get_dataloaders
from src.evaluate import evaluate
from src.utils import class_names
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from datetime import datetime

def train(epochs=10, batch_size=64, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)

    log_dir = f"runs/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tb_writer = SummaryWriter(log_dir=log_dir)
    mlflow.set_experiment("fashion_cnn_experiment")
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.csv", "w", newline="") as f1, \
         open("outputs/train_val_loss.csv", "w", newline="") as f2, \
         open("outputs/val_acc.csv", "w", newline="") as f3:

        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer3 = csv.writer(f3)

        writer1.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        writer2.writerow(["epoch", "train_loss", "val_loss"])
        writer3.writerow(["epoch", "val_acc"])

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

                for name, param in model.named_parameters():
                    tb_writer.add_histogram(f"{name}/weights", param, epoch)
                    if param.grad is not None:
                        tb_writer.add_histogram(f"{name}/grads", param.grad, epoch)

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

            with open("outputs/metrics.csv", "a", newline="") as f1, \
                 open("outputs/train_val_loss.csv", "a", newline="") as f2, \
                 open("outputs/val_acc.csv", "a", newline="") as f3:

                writer1 = csv.writer(f1)
                writer2 = csv.writer(f2)
                writer3 = csv.writer(f3)

                writer1.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_acc])
                writer2.writerow([epoch + 1, avg_train_loss, avg_val_loss])
                writer3.writerow([epoch + 1, val_acc])

            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        model_path = "outputs/model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device, save_dir="outputs", class_names=class_names)
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

        metrics = {"test_loss": test_loss, "test_acc": test_acc}
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("outputs/metrics.json")
        mlflow.log_artifact("outputs/confusion_matrix.csv")
        mlflow.log_artifact("outputs/predictions.csv")
        mlflow.log_artifact("outputs/classification_report.json")
        mlflow.log_artifact("outputs/sample_predictions.png")
        
        img = Image.open("outputs/sample_predictions.png")
        img_tensor = transforms.ToTensor()(img)

        tb_writer.add_scalar("Loss/Test", test_loss, epochs)
        tb_writer.add_scalar("Accuracy/Test", test_acc * 100, epochs)
        tb_writer.add_hparams(
            {
                'lr': lr,
                'batch_size': batch_size,
                'epochs': epochs
            },
            {
                'hparam/accuracy': test_acc,
                'hparam/loss': test_loss
            }
        )
        tb_writer.add_image("Sample Predictions", img_tensor, epochs)
        tb_writer.close()