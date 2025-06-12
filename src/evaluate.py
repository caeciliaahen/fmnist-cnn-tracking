import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import os
import json

def evaluate(model, dataloader, criterion, device, save_dir=None):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        pd.DataFrame(cm, columns=[f"Pred_{i}" for i in range(cm.shape[0])],
                        index=[f"True_{i}" for i in range(cm.shape[0])]) \
            .to_csv(os.path.join(save_dir, "confusion_matrix.csv"))
        pd.DataFrame({"y_true": all_labels, "y_pred": all_preds}).to_csv(
            os.path.join(save_dir, "predictions.csv"), index=False)
        with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=4)

    if not save_dir:
        print(classification_report(all_labels, all_preds))

    return avg_loss, acc, all_preds, all_labels