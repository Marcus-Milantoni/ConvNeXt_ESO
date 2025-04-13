import pandas as pd
import os


patient_data_path = r"D:\Marcus\ESO_DL_DATA\npy_model_in"
clinical_data_scv = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files\Clinical_Data_Eso.xlsx"
csv_saves = r"C:\Users\milantom\OneDrive - LHSC & St. Joseph's\Documents\Eso_deep_learning\csv_files"

def stratify_dataset(df, stratify_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets while preserving the distribution of a specified column.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    stratify_col (str): The column name to stratify by.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame: Training set.
    pd.DataFrame: Testing set.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    
    return train_df, test_df



def main():
    patient_ids = os.listdir(patient_data_path)

    #load in the clinical dataframe
    df = pd.read_excel(clinical_data_scv)
    df.set_index('ID', inplace=True)
    df.drop(columns=['Sex'], inplace=True)
    df.drop(columns=['DOB'], inplace=True)
    df.drop(columns=['Age'], inplace=True)
    df.drop(columns=['PET/CT Date'], inplace=True)
    df.drop(columns=['CT Date'], inplace=True)
    df.drop(columns=['Surgery Date'], inplace=True)
    df.drop(columns=['Primary Procedure'], inplace=True)
    df.drop(columns=['Histologic Type'], inplace=True)

    outcome = {}
    for patient_id, patient_data in df.iterrows():

        if patient_id not in patient_ids:
            df.drop(index=patient_id, inplace=True)
            continue

        if 'pt1' in patient_data['PT Code'].lower() and 'pn0' in patient_data['PN Code'].lower():
            outcome[patient_id] = 1

        else:
            outcome[patient_id] = 0
    
    df['Outcome'] = df.index.map(outcome)

    # Split the dataset into training, validation and testing sets
    train_df, other_df = stratify_dataset(df, 'Outcome', test_size=0.3, random_state=22)
    val_df, test_df = stratify_dataset(other_df, 'Outcome', test_size=0.5, random_state=22)

    train_df.to_csv(os.path.join(csv_saves, 'train.csv'), index=True)
    val_df.to_csv(os.path.join(csv_saves, 'val.csv'), index=True)
    test_df.to_csv(os.path.join(csv_saves, 'test.csv'), index=True)

    # Get the required_paths for each patient in the train_df
    train_dict = {}
    for patient_id, patient_data in train_df.iterrows():
        
        # get the patient_data_paths
        og_paths = {'CT': False, 'PET': False}
        left_paths = {'CT': False, 'PET': False}
        right_paths = {'CT': False, 'PET': False}
        
        patient_path = os.path.join(patient_data_path, patient_id)
        for np_file in os.listdir(patient_path):
            np_path = os.path.join(patient_path, np_file)
            if 'CT_og' in np_file:
                og_paths['CT'] = np_path
            elif 'PET_og' in np_file:
                og_paths['PET'] = np_path
            elif 'CT_left' in np_file:
                left_paths['CT'] = np_path
            elif 'PET_left' in np_file:
                left_paths['PET'] = np_path
            elif 'CT_right' in np_file:
                right_paths['CT'] = np_path
            elif 'PET_right' in np_file:
                right_paths['PET'] = np_path
            else:
                print(f"Unknown file: {np_file}")

        if og_paths['CT'] and og_paths['PET']:
            train_dict[f"{patient_id}_og"] = {
                'CT': og_paths['CT'],
                'PET': og_paths['PET'],
                'Outcome': patient_data['Outcome']
            }
        else:
            print(f"Missing og paths for patient {patient_id}")

        if left_paths['CT'] and left_paths['PET']:
            train_dict[f"{patient_id}_left"] = {
                'CT': left_paths['CT'],
                'PET': left_paths['PET'],
                'Outcome': patient_data['Outcome']
            }
        else:
            print(f"Missing left paths for patient {patient_id}")

        if right_paths['CT'] and right_paths['PET']:
            train_dict[f"{patient_id}_right"] = {
                'CT': right_paths['CT'],
                'PET': right_paths['PET'],
                'Outcome': patient_data['Outcome']
            }
        else:
            print(f"Missing right paths for patient {patient_id}")

    train_path_df = pd.DataFrame.from_dict(train_dict, orient='index')
    train_path_df.to_csv(os.path.join(csv_saves, 'train_paths.csv'), index=True)

    val_dict = {}
    for patient_id, patient_data in val_df.iterrows():
        
        # get the patient_data_paths
        og_paths = {'CT': False, 'PET': False}

        patient_path = os.path.join(patient_data_path, patient_id)
        for npy_file in os.listdir(patient_path):
            np_path = os.path.join(patient_path, npy_file)
            if 'CT_og' in npy_file:
                og_paths['CT'] = np_path
            elif 'PET_og' in npy_file:
                og_paths['PET'] = np_path

            if og_paths['CT'] and og_paths['PET']:
                break

        if og_paths['CT'] and og_paths['PET']:
            val_dict[f"{patient_id}_og"] = {
                'CT': og_paths['CT'],
                'PET': og_paths['PET'],
                'Outcome': patient_data['Outcome']
            }
        else:
            print(f"Missing og paths for patient {patient_id}")

    val_path_df = pd.DataFrame.from_dict(val_dict, orient='index')
    val_path_df.to_csv(os.path.join(csv_saves, 'val_paths.csv'), index=True)


    test_dict = {}
    for patient_id, patient_data in test_df.iterrows():
        
        # get the patient_data_paths
        og_paths = {'CT': False, 'PET': False}

        patient_path = os.path.join(patient_data_path, patient_id)
        for npy_file in os.listdir(patient_path):
            np_path = os.path.join(patient_path, npy_file)
            if 'CT_og' in npy_file:
                og_paths['CT'] = np_path
            elif 'PET_og' in npy_file:
                og_paths['PET'] = np_path

            if og_paths['CT'] and og_paths['PET']:
                break

        if og_paths['CT'] and og_paths['PET']:
            test_dict[f"{patient_id}_og"] = {
                'CT': og_paths['CT'],
                'PET': og_paths['PET'],
                'Outcome': patient_data['Outcome']
            }
        else:
            print(f"Missing og paths for patient {patient_id}")

    
    test_path_df = pd.DataFrame.from_dict(test_dict, orient='index')
    test_path_df.to_csv(os.path.join(csv_saves, 'test_paths.csv'), index=True)

    print("Dataset split and paths saved successfully.")
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")


main()



