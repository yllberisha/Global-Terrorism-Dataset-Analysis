import pandas as pd
from utils import extract_dataset

dataset_file_path = extract_dataset('GlobalTerrorismDataset.zip')
df = pd.read_csv(dataset_file_path, encoding='ISO-8859-1')

print(df.head())