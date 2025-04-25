import torch
from ConvNeXt3D import custom_convnet_33
from sklearn.metrics import roc_curve, accuracy_score, auc
from torch.utils.data import DataLoader
from T1N0_Dataset import Eso_T1N0_Dataset
import pandas as pd
import numpy as np


model_weights = r"D:\Marcus\ESO_DL_model\Model_saves\ConvNeXt_2s\model_epoch_433.pth"
val_csv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\val_paths.csv"
test_csv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\test_paths.csv"

def get_validation_metrics(model):
    model = model
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    val_df = pd.read_csv(val_csv).set_index('ID')
    validation_set = Eso_T1N0_Dataset(val_df)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

    val_outputs = []
    val_outcomes = []
    with torch.no_grad():
        for batch in validation_loader:
            inputs, labels = batch["Data"].to(device), batch["Outcome"].to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            val_outputs.append(outputs.cpu().numpy())
            val_outcomes.append(labels.cpu().numpy())

    val_outputs = np.concatenate(val_outputs, axis=0)
    val_outcomes = np.concatenate(val_outcomes, axis=0)

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

    return val_accuracy, val_auc, fpr_val, tpr_val, thresholds_val


def get_test_metrics(model):
    model = model
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_df = pd.read_csv(test_csv).set_index('ID')
    test_set = Eso_T1N0_Dataset(test_df)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    test_outputs = []
    test_outcomes = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["Data"].to(device), batch["Outcome"].to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            test_outputs.append(outputs.cpu().numpy())
            test_outcomes.append(labels.cpu().numpy())
        
    test_outputs = np.concatenate(test_outputs, axis=0)
    test_outcomes = np.concatenate(test_outcomes, axis=0)

    test_binary_outputs = np.argmax(test_outputs, axis=1).astype(int)
    test_true_labels = np.argmax(test_outcomes, axis=1).astype(int)

    test_outputs_confidence = np.zeros_like(test_binary_outputs).astype(float)
    for index, value in enumerate(test_binary_outputs):
        if value == 1:
            test_outputs_confidence[index] = test_outputs[index][1]
        elif value == 0:
            test_outputs_confidence[index] = 1-test_outputs[index][0]

    test_accuracy = accuracy_score(test_true_labels, test_binary_outputs)
    fpr_test, tpr_test, thresholds_test = roc_curve(test_true_labels, test_outputs_confidence)
    test_auc = auc(fpr_test, tpr_test)

    return test_accuracy, test_auc, fpr_test, tpr_test, thresholds_test


if __name__ == "__main__":
    model = custom_convnet_33(num_classes=2, in_channels=2)
    val_accuracy, val_auc, fpr_val, tpr_val, thresholds_val = get_validation_metrics(model)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    test_accuracy, test_auc, fpr_test, tpr_test, thresholds_test = get_test_metrics(model)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")


