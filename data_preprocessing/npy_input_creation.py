import numpy as np
import os
import torch
import torch.nn.functional as F



resampled_images_npy_path = r""
npy_segmentations_path = r""



def rotate_image(batch, angle):
    """Rotate a set of images by a given angle.
    
    Args:
    batch: torch.Tensor
        A batch of images to be rotated.
    angle: float
        The angle by which to rotate the images.
    
    Returns:
    torch.Tensor
        The rotated images.
    """

    rotation_matrix = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0, 0],
        [torch.sin(angle), torch.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32).unsqueeze(0)

    rotation_matrix = rotation_matrix.repeat(batch.size(0), 1, 1)

    B, D, H, W = batch.shape
    batch = batch.view(B *D, H, W)

    grid = torch.nn.functional.affine_grid(rotation_matrix[:, :2, :], batch.unsqueeze(1).size(), align_corners=True)
    rotated_batch = F.grid_sample(batch.unsqueeze(1), grid, align_corners=False)

    rotated_batch = rotated_batch.view(B, D, H, W)
    return rotated_batch



def randomly_split_patients(list_of_patients, angle_options):
    """
    Randomly split a list of patients into groups based on specified angles.

    Args:
    list_of_patients: list
        A list of patient identifiers.
    angle_options: list
        A list of angles to assign patients to.

    Returns:
    dict: A dictionary where keys are angles and values are lists of patients assigned to those angles.
    """
    angle_dict = {}
    
    num_of_groups = len(angle_options)
    num_of_patients = len(list_of_patients)
    patients_per_group = num_of_patients // num_of_groups

    # Randomly shuffle the list of patients
    np.random.shuffle(list_of_patients)

    # Split the list into groups
    for angle in angle_options:
        angle_dict[angle] = list_of_patients[:patients_per_group]
        list_of_patients = list_of_patients[patients_per_group:]

    # If there are any remaining patients, add them to the last group
    if list_of_patients:
        angle_dict[angle_options[-1]].extend(list_of_patients)

    return angle_dict



def batch_creation(angle, list_of_patients, resampled_images_npy_path, npy_segmentations_path):
    """
    Create batches of images and segmentations for a given angle.

    Args:
    angle: float
        The angle for which to create batches.
    list_of_patients: list
        A list of patient identifiers.
    resampled_images_npy_path: str
        Path to the directory containing resampled images.
    npy_segmentations_path: str
        Path to the directory containing segmentations.

    Returns:
    dict: A dictionary where keys are patient identifiers and values are tensors containing the stacked images and segmentations.
    """

    stack_dicts = {}
    for patient in list_of_patients:
        CT_path = os.path.join(resampled_images_npy_path, patient, f"{patient}_CT.npy")
        PET_path = os.path.join(resampled_images_npy_path, patient, f"{patient}_PET_SUVbw.npy")
        esophagus_seg_path = os.path.join(npy_segmentations_path, patient, f"{patient}_esophagus_segmentation.npy")
        vertebrae_seg_path = os.path.join(npy_segmentations_path, patient, f"{patient}_vertebrae_segmentation.npy")

        esophagus_segmentation = np.load(esophagus_seg_path)
        vertebrae_segmentation = np.load(vertebrae_seg_path)
        CT_npy = np.load(CT_path)
        PET_npy = np.load(PET_path)
        
        # Stack the images and segmentations into a single tensor
        stack = torch.stack((
            torch.from_numpy(PET_npy),
            torch.from_numpy(CT_npy),
            torch.from_numpy(esophagus_segmentation),
            torch.from_numpy(vertebrae_segmentation)
        ))

        stack_dicts[patient] = stack

    return stack_dicts, angle



def unpack_batches(batch):
    """
    Unpack a batch of images and segmentations into individual components.

    Args:
    batch: dict
        A dictionary containing the stacked images and segmentations.

    Returns:
    dict: A dictionary containing unpacked images and segmentations.
    """
    unpacked_batch = {}
    for patient, stack in batch.items():
        PET_npy = stack[0].npy()
        CT_npy = stack[1].npy()
        esophagus_segmentation = stack[2].npy()
        vertebrae_segmentation = stack[3].npy()

        unpacked_batch[patient] = {
            "PET": PET_npy,
            "CT": CT_npy,
            "esophagus_segmentation": esophagus_segmentation,
            "vertebrae_segmentation": vertebrae_segmentation
        }

    return unpacked_batch



def main():
    list_of_patients = os.listdir(resampled_images_npy_path)
    angle_options_1 = [3, 5]
    angle_options_2 = [357, 355]


    randomy_split_patients_1 = randomly_split_patients(list_of_patients, angle_options_1)

    