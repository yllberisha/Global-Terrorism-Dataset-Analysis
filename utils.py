import zipfile
import os

def extract_dataset(zip_file_path):
    csv_file_path = 'globalterrorismdb_0718dist.csv'
    
    if not os.path.exists(csv_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall()
            print(f"Extracted: {zip_ref.namelist()[0]}")
    else:
        print(f"{csv_file_path} already exists!")
    
    return csv_file_path
