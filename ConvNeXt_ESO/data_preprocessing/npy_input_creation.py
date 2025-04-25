import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as img
from tqdm import tqdm
import pandas as pd


resampled_images_npy_path = r"D:\Marcus\ESO_DL_DATA\ESO_DL_DS_resampled"
npy_segmentations_path = r"D:\Marcus\ESO_DL_DATA\ESO_segmentations_resampled"
save_npy_path = r"D:\Marcus\ESO_DL_DATA\npy_model_in"



def find_eso_middle(npy_array: np.ndarray) -> tuple:
    """
    Find the middle of the ESO in a 3D numpy array.
    
    Args:
    npy_array: numpy.ndarray
        A 3D numpy array representing the image.
        
    Returns:
    tuple
        A tuple containing the coordinates of the middle of the ESO in the format (slice, x, y).
    """
    location_arrays = np.where(npy_array == 1)
    front, back = np.min(location_arrays[1]), np.max(location_arrays[1])
    right, left = np.min(location_arrays[2]), np.max(location_arrays[2])
    middle = ((back-front)//2 + front, (left-right)//2 + right)
    
    return middle



def get_upper_t5(npy_array: np.ndarray) -> int:
    """
    Get the upper T5 vertebra location in a 3D numpy array.
    
    Args:
    npy_array: numpy.ndarray
        A 3D numpy array representing the image.
    
    Returns:
    int
        The slice index of the upper T5 vertebra.
    """
    t5_location = np.where(npy_array==1)
    upper_t5 = np.max(t5_location[0])
    
    return upper_t5



def crop_image(npy_img, upper_t5, middle)-> np.ndarray:
    """
    Crop a 3D numpy array image based on the upper T5 vertebra and ESO middle coordinates.
    
    Args:
    npy_img: numpy.ndarray
        A 3D numpy array representing the image.
    upper_t5: int
        The slice index of the upper T5 vertebra.
    middle: tuple
        A tuple containing the coordinates of the middle of the ESO in the format (x, y).

    Returns:
    numpy.ndarray
        The cropped image as a 3D numpy array.
    """
    crop_eso = npy_img[upper_t5-64:upper_t5, middle[0]-64:middle[0]+64, middle[1]-64:middle[1]+64]
    return crop_eso



def rotate_image(npy_array, angle_degrees)-> np.ndarray:
    """
    Rotate a 3D numpy array slice by slice in the XY plane.
    
    Args:
    npy_array: numpy.ndarray
        A 3D numpy array representing the image to be rotated.
    angle_degrees: float
        The angle by which to rotate the image in degrees.

    Returns:
    numpy.ndarray
        The rotated image as a 3D numpy array.
    """
    new_image = np.zeros_like(npy_array)
    for slice_idx in range(npy_array.shape[0]):
        new_image[slice_idx] = img.rotate(npy_array[slice_idx], angle_degrees, reshape=False, order=2)

    return new_image



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



def group_patients(angle_dict: dict, group_size: int = 10) -> dict:
    """
    Groups patients for each angle into smaller subgroups of specified size.

    Args:
        angle_dict (dict): Dictionary with angles as keys and lists of patients as values
        group_size (int): Maximum number of patients per group (default: 10)

    Returns:
        dict: Dictionary with angles as keys and lists of patient groups as values
    """
    for angle, patients in angle_dict.items():
        number_of_groups = (len(patients) + group_size - 1) // group_size
        list_of_groups = []
        
        for i in range(number_of_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, len(patients))
            list_of_groups.append(patients[start_idx:end_idx])

        angle_dict[angle] = list_of_groups

    return angle_dict



def main():
    list_of_patients = os.listdir(resampled_images_npy_path)
    angle_options_right = [3, 5]
    angle_options_left = [357, 355]


    save_paths = {}
    patient_angle_mapping = [] 
    randomy_split_patients_right = randomly_split_patients(list_of_patients, angle_options_right)
    randomly_split_patients_left = randomly_split_patients(list_of_patients, angle_options_left)

    rdm_split_right_grouped = group_patients(randomy_split_patients_right, group_size=10)
    rdm_split_left_grouped = group_patients(randomly_split_patients_left, group_size=10)

    for angle, patient_set in rdm_split_right_grouped.items():
        for group in patient_set:
            for patient in group:
                patient_angle_mapping.append({"patient": patient, "angle": angle, "direction": "right"})

    # Save left angle mappings
    for angle, patient_set in rdm_split_left_grouped.items():
        for group in patient_set:
            for patient in group:
                patient_angle_mapping.append({"patient": patient, "angle": angle, "direction": "left"})

    # Save the mapping to a CSV file
    patient_angle_mapping_df = pd.DataFrame(patient_angle_mapping)
    patient_angle_mapping_df.to_csv(os.path.join(save_npy_path, "patient_angle_mapping.csv"), index=False)


    #Process the patients for the right angles
    for angle, patient_set in tqdm(rdm_split_right_grouped.items()):
        for patient_group in tqdm(patient_set):
            for patient in tqdm(patient_group):

                if not os.path.exists(os.path.join(save_npy_path, patient)):
                    os.makedirs(os.path.join(save_npy_path, patient))

                #generate save paths
                og_ct_save_path = os.path.join(save_npy_path, patient, f"{patient}_CT_og.npy")
                og_pet_save_path = os.path.join(save_npy_path, patient, f"{patient}_PET_og.npy")
                rotated_ct_save_path = os.path.join(save_npy_path, patient, f"{patient}_CT_right_{angle}.npy")
                rotated_pet_save_path = os.path.join(save_npy_path, patient, f"{patient}_PET_right_{angle}.npy")

                # load the arrays for the patient
                CT_array = np.load(os.path.join(resampled_images_npy_path, patient, f"{patient}_CT.npy"))
                PET_aray = np.load(os.path.join(resampled_images_npy_path, patient, f"{patient}_PET_SUVbw.npy"))
                T5_array = np.load(os.path.join(npy_segmentations_path, patient, f"{patient}_vertebrae_T5.npy"))
                eso_array = np.load(os.path.join(npy_segmentations_path, patient, f"{patient}_esophagus.npy"))

                #find eso middle and upper t5
                eso_middle = find_eso_middle(eso_array)
                upper_t5 = get_upper_t5(T5_array)

                #crop the images
                cropped_CT = crop_image(CT_array, upper_t5, eso_middle)
                cropped_PET = crop_image(PET_aray, upper_t5, eso_middle)

                # save the cropped images
                np.save(og_ct_save_path, cropped_CT)
                np.save(og_pet_save_path, cropped_PET)

                del cropped_CT, cropped_PET, eso_middle, upper_t5


                # Now do the same for the rotated images
                rotated_ct = rotate_image(CT_array, angle)
                rotated_pet = rotate_image(PET_aray, angle)
                rotated_t5 = rotate_image(T5_array, angle)
                rotated_eso = rotate_image(eso_array, angle)

                eso_middle = find_eso_middle(rotated_eso)
                upper_t5 = get_upper_t5(rotated_t5)

                cropped_rotated_CT = crop_image(rotated_ct, upper_t5, eso_middle)
                cropped_rotated_PET = crop_image(rotated_pet, upper_t5, eso_middle)

                np.save(rotated_ct_save_path, cropped_rotated_CT)
                np.save(rotated_pet_save_path, cropped_rotated_PET)
                del rotated_ct, rotated_pet, rotated_t5, rotated_eso, eso_middle, upper_t5, cropped_rotated_CT, cropped_rotated_PET, CT_array, PET_aray, T5_array, eso_array

                save_paths[patient] = {
                    "og_ct": og_ct_save_path,
                    "og_pet": og_pet_save_path,
                    "right_ct": rotated_ct_save_path,
                    "right_pet": rotated_pet_save_path
                }

                del og_ct_save_path, og_pet_save_path, rotated_ct_save_path, rotated_pet_save_path

            pd.DataFrame.from_dict(save_paths, orient='index').to_csv(os.path.join(save_npy_path, f"in_save_paths.csv"))


    #Process the patients for the left angles
    for angle, patient_set in tqdm(rdm_split_left_grouped.items()):
        for patient_group in tqdm(patient_set):
            for patient in tqdm(patient_group):

                if not os.path.exists(os.path.join(save_npy_path, patient)):
                    os.makedirs(os.path.join(save_npy_path, patient))

                #generate save paths
                og_ct_save_path = os.path.join(save_npy_path, patient, f"{patient}_CT_og.npy")
                og_pet_save_path = os.path.join(save_npy_path, patient, f"{patient}_PET_og.npy")
                rotated_ct_save_path = os.path.join(save_npy_path, patient, f"{patient}_CT_left_{angle}.npy")
                rotated_pet_save_path = os.path.join(save_npy_path, patient, f"{patient}_PET_left_{angle}.npy")

                # load the arrays for the patient
                CT_array = np.load(os.path.join(resampled_images_npy_path, patient, f"{patient}_CT.npy"))
                PET_aray = np.load(os.path.join(resampled_images_npy_path, patient, f"{patient}_PET_SUVbw.npy"))
                T5_array = np.load(os.path.join(npy_segmentations_path, patient, f"{patient}_vertebrae_T5.npy"))
                eso_array = np.load(os.path.join(npy_segmentations_path, patient, f"{patient}_esophagus.npy"))

                #find eso middle and upper t5
                eso_middle = find_eso_middle(eso_array)
                upper_t5 = get_upper_t5(T5_array)

                #crop the images
                cropped_CT = crop_image(CT_array, upper_t5, eso_middle)
                cropped_PET = crop_image(PET_aray, upper_t5, eso_middle)

                # save the cropped images
                np.save(og_ct_save_path, cropped_CT)
                np.save(og_pet_save_path, cropped_PET)

                del cropped_CT, cropped_PET, eso_middle, upper_t5


                # Now do the same for the rotated images
                rotated_ct = rotate_image(CT_array, angle)
                rotated_pet = rotate_image(PET_aray, angle)
                rotated_t5 = rotate_image(T5_array, angle)
                rotated_eso = rotate_image(eso_array, angle)

                eso_middle = find_eso_middle(rotated_eso)
                upper_t5 = get_upper_t5(rotated_t5)

                cropped_rotated_CT = crop_image(rotated_ct, upper_t5, eso_middle)
                cropped_rotated_PET = crop_image(rotated_pet, upper_t5, eso_middle)

                np.save(rotated_ct_save_path, cropped_rotated_CT)
                np.save(rotated_pet_save_path, cropped_rotated_PET)
                del rotated_ct, rotated_pet, rotated_t5, rotated_eso, eso_middle, upper_t5, cropped_rotated_CT, cropped_rotated_PET, CT_array, PET_aray, T5_array, eso_array

                save_paths[patient] = {
                    "left_ct": rotated_ct_save_path,
                    "left_pet": rotated_pet_save_path
                }

                del og_ct_save_path, og_pet_save_path, rotated_ct_save_path, rotated_pet_save_path

            pd.DataFrame.from_dict(save_paths, orient='index').to_csv(os.path.join(save_npy_path, f"in_save_paths.csv"))



if __name__ == "__main__":
    main()