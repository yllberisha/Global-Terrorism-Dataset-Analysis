import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import extract_dataset

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset_file_path = extract_dataset('GlobalTerrorismDataset.zip')
df = pd.read_csv(dataset_file_path, encoding='ISO-8859-1')

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
     print(f'Missing columns: {missing_columns}')

# --- Selection Of The Subset Of Attributes ---
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

# --- Sample Selection ---
filtered_df['Decade'] = (filtered_df['Year'] // 10) * 10

sampling_fraction = 0.1

# Sample 10% of each group by Decade and Region
sampled_df = filtered_df.groupby(['Decade', 'Region']).apply(lambda x: x.sample(frac=sampling_fraction, random_state=1)).reset_index(drop=True)

output_file_path = 'Filtered_GlobalTerrorismDataset.csv'
filtered_df.to_csv(output_file_path, index=False)

print(f'Data saved to {output_file_path} with selected columns and new names.')

# --- Define Data Types ---
attribute_classification = {
    'Nominal': [
        'country_txt', 'region_txt', 'city', 'attacktype1_txt', 'targtype1_txt', 
        'natlty1_txt', 'gname', 'weaptype1_txt', 'dbsource'
    ],
    'Ordinal': [
        'success', 'suicide'
    ],
    'Interval': [
        'iyear', 'imonth', 'iday', 'resolution'
    ],
    'Ratio': [
        'nkill', 'nwound', 'nkillus', 'nwoundus', 'nperps'
    ]
}

print("Type of Attributes Classification:")
for attribute_type, columns in attribute_classification.items():
    print(f"\n{attribute_type} Attributes:")
    for col in columns:
        print(f" - {col}")

# --- Discretization ---
bins = [0, 10, 100, 500, 1000, 1570]
labels = ['0-10', '11-100', '101-500', '501-1000', '1001-1570']
df['victim_range'] = pd.cut(df['nkill'], bins=bins, labels=labels, right=True)

df_cleaned = df.dropna(subset=['nkill', 'victim_range'])

victim_distribution = df_cleaned['victim_range'].value_counts()
print(f'\nBinning:\n{victim_distribution}')

# Decade Distribution
bins_decades = [1970, 1980, 1990, 2000, 2010, 2020]  
labels_decades = ['1970s', '1980s', '1990s', '2000s', '2010s']  
df['decade'] = pd.cut(df['iyear'], bins=bins_decades, labels=labels_decades, right=False)

df_cleaned = df.dropna(subset=['nkill', 'decade'])

decade_distribution = df_cleaned['decade'].value_counts()
print(f'\nDecade Distribution:\n{decade_distribution}')

# --- Dimension Reduction ---
df['attacktype1'] = df['attacktype1'].replace('', np.nan).fillna('Unknown')
df['weaptype1'] = df['weaptype1'].replace('', np.nan).fillna('Unknown')

# One-hot encode only the 'attacktype1' and 'weaptype1' columns
attack_weap_dummies = pd.get_dummies(df[['attacktype1', 'weaptype1']], drop_first=True)

other_features = df[['nperps', 'nkill', 'suicide', 'success']]
features = pd.concat([other_features, attack_weap_dummies], axis=1)
features = features.applymap(lambda x: np.nan if x < 0 else x)
features.fillna(features.median(), inplace=True)

scaler = StandardScaler()
features[['nperps', 'nkill']] = scaler.fit_transform(features[['nperps', 'nkill']])
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
reduced_df = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])

print(pca.explained_variance_ratio_)