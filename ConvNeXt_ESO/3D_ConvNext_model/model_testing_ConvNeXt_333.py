import torch
from ConvNeXt3D import custom_convnet_333
from sklearn.metrics import roc_curve, accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from T1N0_Dataset import Eso_T1N0_Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


model_weights = r"D:\Marcus\ESO_DL_model\Model_saves\ConvNeXt_333\best_val_auc.pth"
val_csv = r"C:\Users\wanged\Downloads\val_paths.csv"
test_csv = r"C:\Users\wanged\Downloads\test_paths.csv"
train_csv = r"C:\Users\wanged\Downloads\train_paths.csv"

def get_validation_metrics(model):
    model = model
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    device = "cpu"
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
    conf_matrix = confusion_matrix(val_true_labels, val_binary_outputs)

    return val_accuracy, val_auc, fpr_val, tpr_val, thresholds_val, conf_matrix


def get_test_metrics(model):
    model = model
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    device = "cpu"
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
    conf_matrix = confusion_matrix(test_true_labels, test_binary_outputs)

    return test_accuracy, test_auc, fpr_test, tpr_test, thresholds_test, conf_matrix

def get_train_metrics(model):
    model = model
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    device = "cpu"
    model.to(device)

    train_df = pd.read_csv(train_csv).set_index('ID')
    train_set = Eso_T1N0_Dataset(train_df)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=4)

    train_outputs = []
    train_outcomes = []
    with torch.no_grad():
        for batch in train_loader:
            inputs, labels = batch["Data"].to(device), batch["Outcome"].to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            train_outputs.append(outputs.cpu().numpy())
            train_outcomes.append(labels.cpu().numpy())

    train_outputs = np.concatenate(train_outputs, axis=0)
    train_outcomes = np.concatenate(train_outcomes, axis=0)

    train_binary_outputs = np.argmax(train_outputs, axis=1).astype(int)
    train_true_labels = np.argmax(train_outcomes, axis=1).astype(int)

    train_outputs_confidence = np.zeros_like(train_binary_outputs).astype(float)
    for index, value in enumerate(train_binary_outputs):
        if value == 1:
            train_outputs_confidence[index] = train_outputs[index][1]
        elif value == 0:
            train_outputs_confidence[index] = 1-train_outputs[index][0]

    train_accuracy = accuracy_score(train_true_labels, train_binary_outputs)
    fpr_train, tpr_train, thresholds_train = roc_curve(train_true_labels, train_outputs_confidence)
    train_auc = auc(fpr_train, tpr_train)
    conf_matrix = confusion_matrix(train_true_labels, train_binary_outputs)

    return train_accuracy, train_auc, fpr_train, tpr_train, thresholds_train, conf_matrix


def plot_confusion_matrix(cm, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["All Other", "T1N0 Cancer"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    model = custom_convnet_333(num_classes=2, in_channels=2)
    val_accuracy, val_auc, fpr_val, tpr_val, thresholds_val, confusion_matrix_val = get_validation_metrics(model)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    test_accuracy, test_auc, fpr_test, tpr_test, thresholds_test, confusion_matrix_test = get_test_metrics(model)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    train_accuracy, train_auc, fpr_train, tpr_train, thresholds_train, confusion_matrix_train = get_train_metrics(model)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train AUC: {train_auc:.4f}")

    plot_confusion_matrix(confusion_matrix_val, "Validation Confusion Matrix")
    plot_confusion_matrix(confusion_matrix_test, "Test Confusion Matrix")
    plot_confusion_matrix(confusion_matrix_train, "Train Confusion Matrix")

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr_val, tpr_val, label=f'Validation ROC curve (area = {val_auc:.4f})')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (area = {test_auc:.4f})')
    plt.plot(fpr_train, tpr_train, label=f'Train ROC curve (area = {train_auc:.4f})')
    plt.title('Receiver Operating Characteristic Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')
    plt.show()


