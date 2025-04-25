import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score, auc
from T1N0_Dataset import Eso_T1N0_Dataset
from train_and_val_epoch import train_one_epoch, val_one_epoch
from ConvNeXt3D import custom_convnet_33
import os


train_csv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\train_paths.csv"
val_csv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\val_paths.csv"
log_path = r"D:\Marcus\ESO_DL_DATA\DL_model_logs\ConvNeXt_2s"
model_save_path = r"D:\Marcus\ESO_DL_model\Model_saves\ConvNeXt_2s"

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

train_df = pd.read_csv(train_csv).set_index('ID')
val_df = pd.read_csv(val_csv).set_index('ID')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(log_dir=log_path)

    training_set = Eso_T1N0_Dataset(train_df)
    validation_set = Eso_T1N0_Dataset(val_df)

    training_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

    model = custom_convnet_33(num_classes=2, in_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999))
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)


    num_epochs = 500
    best_v_AUC = 0.0
    best_v_loss = 1e10
    last_5_auc = []

    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
        train_loss, train_outputs, train_outcomes = train_one_epoch(model, optimizer, loss_function, training_loader, device)
    
        train_binary_outputs = np.argmax(train_outputs, axis=1).astype(int) 
        train_true_labels = np.argmax(train_outcomes, axis=1).astype(int)

        #Get the model confidence
        train_outputs_confidence = np.zeros_like(train_binary_outputs).astype(float)
        for index, value in enumerate(train_binary_outputs):
            if value == 1:
                train_outputs_confidence[index] = train_outputs[index][1]
            elif value == 0:
                train_outputs_confidence[index] = 1-train_outputs[index][0]

        # Calculate metrics
        train_accuracy = accuracy_score(train_true_labels, train_binary_outputs)
        fpr_train, tpr_train, thresholds_train = roc_curve(train_true_labels, train_outputs_confidence)
        train_auc = auc(fpr_train, tpr_train)

        # Log the training_metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("AUC/train", train_auc, epoch)
        writer.add_pr_curve("Train Precision-Recall", train_outcomes, train_outputs, epoch)

        # Validation
        val_loss, val_outputs, val_outcomes = val_one_epoch(model, loss_function, validation_loader, device)

        val_binary_outputs = np.argmax(val_outputs, axis=1).astype(int)
        val_true_labels = np.argmax(val_outcomes, axis=1).astype(int)

        val_outputs_confidence = np.zeros_like(val_binary_outputs).astype(float)
        for index, value in enumerate(val_binary_outputs):
            if value == 1:
                val_outputs_confidence[index] = val_outputs[index][1]
            elif value == 0:
                val_outputs_confidence[index] = 1-val_outputs[index][0]

        val_accuracy = accuracy_score(val_true_labels, val_binary_outputs)
        fpr_val, tpr_val, thresholds_val = roc_curve(val_true_labels, val_outputs_confidence)
        val_auc = auc(fpr_val, tpr_val)

        if len(last_5_auc) < 5:
            last_5_auc.append(val_auc)
        else:
            last_5_auc.pop(0)
            last_5_auc.append(val_auc)

        # Log the validation metrics
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_pr_curve("Validation Precision-Recall", val_outcomes, val_outputs, epoch)


        # Save the best model based on validation loss
        if val_loss < best_v_loss:
            best_v_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_val_loss.pth"))

        if val_auc > best_v_AUC:
            best_v_AUC = val_auc
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_val_auc.pth"))

        # Save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f"model_epoch_{epoch}.pth"))

        if val_auc < any(last_5_auc[:-1]):
            torch.save(model.state_dict(), os.path.join(model_save_path, f"model_epoch_{epoch}.pth"))            

        scheduler.step()
    
    writer.close()

if __name__ == "__main__":
    main()
