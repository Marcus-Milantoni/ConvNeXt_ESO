import os 
import numpy as np
from tqdm import tqdm

patient_numpy_paths = r"D:\Marcus\ESO_DL_DATA\npy_model_in"


def data_standardization(npy_array: np.ndarray) -> np.ndarray:
    """
    Standardizes the input numpy array to have a mean of 0 and a standard deviation of 1.
    
    Parameters:
    npy_array (np.ndarray): Input numpy array to be standardized.
    
    Returns:
    np.ndarray: Standardized numpy array.
    """
    mean = np.mean(npy_array)
    std = np.std(npy_array)
    
    standardized_array = (npy_array - mean) / std
    return standardized_array

def main(patient_numpy_paths: str):
    """
    Main function to iterate through all numpy files in the specified directory,
    standardize them, and save the standardized arrays back to their respective files.

    Parameters:
    patient_numpy_paths (str): Path to the directory containing patient numpy files.
    """
    for patient_id in tqdm(os.listdir(patient_numpy_paths)):
        patient_path = os.path.join(patient_numpy_paths, patient_id)
        if not os.path.isdir(patient_path):
            continue
        
        for npy_file in tqdm(os.listdir(patient_path)):
            if npy_file.endswith('.npy'):
                npy_file_path = os.path.join(patient_path, npy_file)
                npy_array = np.load(npy_file_path)
                standardized_array = data_standardization(npy_array)
                # Save the standardized array back to the file
                np.save(npy_file_path, standardized_array)


main(patient_numpy_paths=patient_numpy_paths)
