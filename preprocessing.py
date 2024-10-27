import pandas as pd
from utils import extract_dataset

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset_file_path = extract_dataset('GlobalTerrorismDataset.zip')
df = pd.read_csv(dataset_file_path, encoding='ISO-8859-1')

selected_columns = [
    'iyear', 'imonth', 'iday', 'extended', 'resolution', 'country_txt', 
    'region_txt', 'city', 'success', 'suicide', 'attacktype1_txt', 
    'targtype1_txt', 'natlty1_txt', 'gname', 'nperps', 'weaptype1_txt', 
    'nkill', 'nwound', 'nkillus', 'nwoundus', 'dbsource'
]

filtered_df = df[selected_columns]

column_renaming = {
    'iyear': 'Year', 'imonth': 'Month', 'iday': 'Day', 'extended': 'Extended', 
    'resolution': 'Resolution', 'country_txt': 'Country', 'region_txt': 'Region', 
    'city': 'City', 'success': 'Success', 'suicide': 'Suicide', 
    'attacktype1_txt': 'Attack Type', 'targtype1_txt': 'Target Type', 
    'natlty1_txt': 'Attackers Nationality', 'gname': 'Group Name', 
    'nperps': 'Number of Terrorists', 'weaptype1_txt': 'Weapon Type', 
    'nkill': 'Number of Killed People', 'nwound': 'Number of Wounded People', 
    'nkillus': 'Number of Killed US People', 'nwoundus': 'Number of Wounded US People', 
    'dbsource': 'Database Source'
}

filtered_df = filtered_df.rename(columns=column_renaming)

output_file_path = 'Filtered_GlobalTerrorismDataset.csv'
filtered_df.to_csv(output_file_path, index=False)

print(f'Data saved to {output_file_path} with selected columns and new names.')

# --- Define Data Types ---
attribute_classification = {
    'Nominal': [
        'eventid', 'country_txt', 'region_txt', 'provstate', 'city', 'specificity',
        'location', 'summary', 'attacktype1_txt', 'attacktype2_txt', 'attacktype3_txt',
        'targtype1_txt', 'targsubtype1_txt', 'targtype2_txt', 'targsubtype2_txt', 
        'targtype3_txt', 'targsubtype3_txt', 'target1', 'target2', 'target3', 
        'gname', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 
        'motive', 'weaptype1_txt', 'weapsubtype1_txt', 'weaptype2_txt', 
        'weapsubtype2_txt', 'weaptype3_txt', 'weapsubtype3_txt', 'weaptype4_txt', 
        'weapsubtype4_txt', 'claimmode_txt', 'claimmode2_txt', 'claimmode3_txt', 
        'compclaim', 'property', 'propextent_txt', 'propcomment', 'divert', 
        'kidhijcountry', 'ransomnote', 'hostkidoutcome_txt', 'addnotes', 'scite1', 
        'scite2', 'scite3', 'dbsource', 'related'
    ],
    'Ordinal': [
        'doubtterr', 'multiple', 'success', 'suicide', 'ishostkid', 'ransom', 
        'claimed', 'claim2', 'claim3'
    ],
    'Interval': [
        'iyear', 'imonth', 'iday', 'resolution', 'latitude', 'longitude'
    ],
    'Ratio': [
        'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 
        'propvalue', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 
        'hostkidoutcome'
    ]
}

print("Type of Attributes Classification:")
for attribute_type, columns in attribute_classification.items():
    print(f"\n{attribute_type} Attributes:")
    for col in columns:
        print(f" - {col}")

# --- DATA QUALITY ---
# Completeness
missing_values = df.isnull().sum()
completeness_percentage = (1 - (missing_values / len(df))) * 100

completeness_analysis = pd.DataFrame({
    'Missing Values Count': missing_values,
    'Completeness Percentage': completeness_percentage
})

print('\nCompleteness Analysis:\n', completeness_analysis)

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

