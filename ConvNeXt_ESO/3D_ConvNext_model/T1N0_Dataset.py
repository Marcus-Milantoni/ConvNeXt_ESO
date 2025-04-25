import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np
import os
import pandas as pd



class Eso_T1N0_Dataset(Dataset):
    "Esophagus dataset for classification of T1N0 cancers in medical images (PET/CT)"

    def __init__ (self, dataframe: pd.DataFrame, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing patient IDs and outcomes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.df = dataframe
        self.patient_IDs = self.df.index.to_list()
        self.transform = transform

    def __len__(self):
        return len(self.patient_IDs)
    
    def __getitem__ (self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_ID = self.patient_IDs[idx]
        PET_npy = np.load(self.df.loc[patient_ID, 'PET']).astype(np.float32)
        CT_npy = np.load(self.df.loc[patient_ID, 'CT']).astype(np.float32)
        outcome = np.array(self.df.loc[patient_ID,'Outcome']).astype(np.float32)
        outcome = torch.nn.functional.one_hot(torch.tensor(outcome, dtype=torch.long), num_classes=2).float()

        stack = torch.stack((
                torch.from_numpy(PET_npy),
                torch.from_numpy(CT_npy),
            ))

        return {"Data": stack, "Outcome": outcome}
    



class Eso_T_Dataset(Dataset):
    "Esophagus dataset for classification of thr T stage of cancer in medical images (PET/CT)"

    def __init__ (self, patient_IDs, outcome_pT, root_dir, transform=None):
        """
        Args:
            patient_IDs (list): List of patient IDs to be included in the dataset.
            outcome_pT (list): List of T-stage outcomes for each patient ID.
            root_dir (string): Directory with all the images and contours (PET, CT, segmentation).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.patient_IDs = patient_IDs
        self.root_dir = root_dir
        self.outcome_pT = outcome_pT
        self.transform = transform

    def __len__(self):
        return len(self.patient_IDs)
    
    def __getitem__ (self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_ID = self.patient_IDs[idx]
        PET_npy = np.load(os.path.join(self.root_dir, patient_ID , "PET.npy"))
        CT_npy = np.load(os.path.join(self.root_dir, patient_ID , "CT.npy"))
        Segmentation_npy = np.load(os.path.join(self.root_dir, patient_ID , "segmentation.npy")).astype(np.float32)
        outcome = np.array(self.outcome_pT[idx]).astype(np.float32)


        stack = torch.stack((
                torch.from_numpy(PET_npy),
                torch.from_numpy(CT_npy),
                torch.from_numpy(Segmentation_npy)
            ))

        return {"Data": stack, "outcome": torch.tensor(outcome)}




class Eso_N_Dataset(Dataset):
    "Esophagus dataset for classification of the N stage of cancer in medical images (PET/CT)"

    def __init__ (self, patient_IDs, outcome_pN, root_dir, transform=None):
        """
        Args:
            patient_IDs (list): List of patient IDs to be included in the dataset.
            outcome_pN (list): List of N-stage outcomes for each patient ID.
            root_dir (string): Directory with all the images and contours (PET, CT, segmentation).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.patient_IDs = patient_IDs
        self.root_dir = root_dir
        self.outcome_pN = outcome_pN
        self.transform = transform

    def __len__(self):
        return len(self.patient_IDs)
    
    def __getitem__ (self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_ID = self.patient_IDs[idx]
        PET_npy = np.load(os.path.join(self.root_dir, patient_ID , "PET.npy"))
        CT_npy = np.load(os.path.join(self.root_dir, patient_ID , "CT.npy"))
        Segmentation_npy = np.load(os.path.join(self.root_dir, patient_ID , "segmentation.npy")).astype(np.float32)
        outcome = np.array(self.outcome_pN[idx]).astype(np.float32)


        stack = torch.stack((
                torch.from_numpy(PET_npy),
                torch.from_numpy(CT_npy),
                torch.from_numpy(Segmentation_npy)
            ))

        return {"Data": stack, "outcome": torch.tensor(outcome)}
    



class Eso_TN_Dataset(Dataset):
    "Esophagus dataset for classification of T stage and N stage of cancer in medical images (PET/CT) (for a multi-learn model)"

    def __init__ (self, patient_IDs, outcome_pT, outcome_pN, root_dir, transform=None):
        """
        Args:
            patient_IDs (list): List of patient IDs to be included in the dataset.
            outcomes_pT (list): List of T-stage outcomes for each patient ID.
            outcomes_pN (list): List of N-stage outcomes for each patient ID.
            root_dir (string): Directory with all the images and contours (PET, CT, segmentation).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.patient_IDs = patient_IDs
        self.root_dir = root_dir
        self.outcome_pT = outcome_pT
        self.outcome_pN = outcome_pN
        self.transform = transform

    def __len__(self):
        return len(self.patient_IDs)
    
    def __getitem__ (self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_ID = self.patient_IDs[idx]
        PET_npy = np.load(os.path.join(self.root_dir, patient_ID , "PET.npy"))
        CT_npy = np.load(os.path.join(self.root_dir, patient_ID , "CT.npy"))
        Segmentation_npy = np.load(os.path.join(self.root_dir, patient_ID , "segmentation.npy")).astype(np.float32)
        outcome_pT = np.array(self.outcome_pT[idx]).astype(np.float32)
        outcome_pN = np.array(self.outcome_pN[idx]).astype(np.float32)


        stack = torch.stack((
                torch.from_numpy(PET_npy),
                torch.from_numpy(CT_npy),
                torch.from_numpy(Segmentation_npy)
            ))

        return {"Data": stack, "outcome_pT": torch.tensor(outcome_pT), "outcome_pN": torch.tensor(outcome_pN)}
