import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score, auc
from T1N0_Dataset import Eso_T1N0_Dataset
from train_and_val_epoch import train_one_epoch, val_one_epoch
from ConvNeXt3D import convNeXt_tiny_ESO
import os


train_csv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\train_paths.csv"
val_csv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\val_paths.csv"
log_path = r"D:\Marcus\ESO_DL_model\DL_model_logs\ConvNeXt_Tiny"
model_save_path = r"D:\Marcus\ESO_DL_model\Model_saves\ConvNeXt_Tiny"

train_df = pd.read_csv(train_csv).set_index('ID')
val_df = pd.read_csv(val_csv).set_index('ID')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(log_dir=log_path)

    training_set = Eso_T1N0_Dataset(train_df)
    validation_set = Eso_T1N0_Dataset(val_df)

    training_loader = DataLoader(training_set, batch_size=8, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_set, batch_size=8, shuffle=False, num_workers=4)

    model = convNeXt_tiny_ESO(num_classes=2, in_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loss_function = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    num_epochs = 500
    best_AUC = 0.0
    best_v_loss = 1e10

    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
        train_loss, train_outputs, train_outcomes = train_one_epoch(model, optimizer, loss_function, training_loader, device)

        # Calculate metrics
        train_accuracy = accuracy_score(train_outcomes, np.round(train_outputs))
        fpr_train, tpr_train, thresholds_train = roc_curve(train_outcomes, train_outputs)
        train_auc = auc(fpr_train, tpr_train)

        # Log the training_metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("AUC/train", train_auc, epoch)
        writer.add_pr_curve("Train Precision-Recall", train_outcomes, train_outputs, epoch)

        # Validation
        val_loss, val_outputs, val_outcomes = val_one_epoch(model, loss_function, validation_loader, device)

        val_accuracy = accuracy_score(val_outcomes, np.round(val_outputs))
        fpr_val, tpr_val, thresholds_val = roc_curve(val_outcomes, val_outputs)
        val_auc = auc(fpr_val, tpr_val)

        # Log the validation metrics
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_pr_curve("Validation Precision-Recall", val_outcomes, val_outputs, epoch)


        # Save the best model based on validation loss
        if val_loss < best_v_loss:
            best_v_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_val_loss.pth"))

        # Save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f"model_epoch_{epoch}.pth"))
    
    writer.close()

main()
