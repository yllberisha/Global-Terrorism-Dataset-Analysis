import pandas as pd
from utils import extract_dataset

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset_file_path = extract_dataset('GlobalTerrorismDataset.zip')
df = pd.read_csv(dataset_file_path, encoding='ISO-8859-1')

# --- Define Data Types ---
# TODO: Define Data Types

# --- DATA QUALITY ---
# Completeness
missing_values = df.isnull().sum()
completeness_percentage = (1 - (missing_values / len(df))) * 100

completeness_analysis = pd.DataFrame({
    'Missing Values Count': missing_values,
    'Completeness Percentage': completeness_percentage
})

print('Completeness Analysis:\n', completeness_analysis)

columns_50p_missing = completeness_analysis[completeness_analysis['Completeness Percentage'] < 50].index
if not columns_50p_missing.empty:
    print(f'\nColumns with More Than 50% Missing Values:\n{columns_50p_missing.tolist()}')
else:
    print('No columns have more than 50% missing values.')

# Uniqueness
unique_counts = df.nunique()

uniqueness_analysis = pd.DataFrame({
    'Unique Values Count': unique_counts,
})

print('\nUniqueness Analysis:\n', uniqueness_analysis)

duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f'\nThere are {duplicate_count} duplicate rows in the dataset.')
else:
    print('No duplicate rows found in the dataset.')

# Accuracy Checks
invalid_kill_counts = df[df['nkill'] < 0]
if not invalid_kill_counts.empty:
    print(f'\nInvalid kill counts found:\n{invalid_kill_counts[["eventid", "nkill"]]}')

# Check for missing values in year, month, and day
required_columns = ['iyear', 'imonth', 'iday']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f'Missing columns: {missing_columns}')

