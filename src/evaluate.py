# File: src/evaluate.py

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def evaluate(model, dataloader, criterion, device, save_dir=None, class_names=None):
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
        
        sample_images = torch.stack([dataloader.dataset[i][0] for i in range(8)])
        sample_labels = [dataloader.dataset[i][1] for i in range(8)]
        sample_preds = model(sample_images.to(device)).argmax(1).cpu().tolist()
        plot_sample_predictions(sample_images, sample_labels, sample_preds, class_names,
                                save_path=os.path.join(save_dir, "sample_predictions.png"))

    if not save_dir:
        print(classification_report(all_labels, all_preds))

    return avg_loss, acc, all_preds, all_labels

def plot_sample_predictions(images, labels, preds, class_names, save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            break
        img = images[idx].cpu().numpy()
        img = img / 2 + 0.5  # unnormalize
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img.squeeze(), cmap='gray')

        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        color = 'green' if labels[idx] == preds[idx] else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()