import os
from totalsegmentator.python_api import totalsegmentator
import torch
from tqdm import tqdm
import pandas as pd

data_path = r"D:\Marcus\Dataset_for_eso_dl"
output_path = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\ESO_segmentations"

if __name__ == '__main__':
    try:

        #Iterate over the rows of the dataframe (all of the patients in the dataset)
        for patient_ID in  tqdm(os.listdir(data_path)):
           
            if "Studies" in patient_ID:
                continue
            
            pt_number = patient_ID.split("_")[1].split("O")[1]
            if int(pt_number) < 0:
                continue

            for scan in os.listdir(os.path.join(data_path, patient_ID)):
                if "PET" in scan:
                    continue
                
                ct_path = os.path.join(data_path, patient_ID, scan)
                    
                torch.cuda.empty_cache()
                print(f'Processing patient {patient_ID}')

                #Create the output path for the segmentations

                out_path = os.path.join(output_path, patient_ID)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                # Run the segmentation
                totalsegmentator(ct_path, out_path, device='gpu', output_type='dicom', ml=True, roi_subset=['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'esophagus', 'vertebrae_T5', 'stomach'], body_seg=True)

    except Exception as e:
        print(e)

