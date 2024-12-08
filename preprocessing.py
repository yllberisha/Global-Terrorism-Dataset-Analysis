import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from utils import extract_dataset
from datetime import datetime
import subprocess
import os
from scipy.stats import t
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if not os.path.exists('Preprocessed_GDP_Dataset.csv'):
    subprocess.run(["python3", "preprocess_gdp_dataset.py"])

dataset_file_path = extract_dataset('GlobalTerrorismDataset.zip')
df = pd.read_csv(dataset_file_path, encoding='ISO-8859-1')

# --- DATA COLLECTION ---
db_source_counts = df['dbsource'].value_counts(normalize=True) * 100

top_4_db_sources = db_source_counts.head(4).to_frame(name='Contribution Percentage')
others_percentage = pd.DataFrame({'Contribution Percentage': [db_source_counts.iloc[4:].sum()]}, index=['Others'])

db_sources_summary = pd.concat([top_4_db_sources, others_percentage])
db_sources_summary.index.name = 'Database Source'

print("Database Sources:\n")
print(db_sources_summary.to_string(float_format="{:,.6f}".format))

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
    print('\nNo duplicate rows found in the dataset.\n')

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

print(f'\n\nSelected Columns:\n{selected_columns}\n')

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

filtered_df = df[selected_columns]

column_renaming = {
    'iyear': 'Year', 'imonth': 'Month', 'iday': 'Day', 'extended': 'Extended', 
    'resolution': 'Resolution', 'country_txt': 'Country', 'region_txt': 'Region', 
    'city': 'City', 'success': 'Success', 'suicide': 'Suicide', 
    'attacktype1_txt': 'Attack Type', 'targtype1_txt': 'Target Type', 
    'natlty1_txt': 'Target Nationality', 'gname': 'Attacking Group Name', 
    'nperps': 'Number of Terrorists', 'weaptype1_txt': 'Weapon Type', 
    'nkill': 'Number of Killed People', 'nwound': 'Number of Wounded People', 
    'nkillus': 'Number of Killed US People', 'nwoundus': 'Number of Wounded US People', 
    'dbsource': 'Database Source'
}

filtered_df = filtered_df.rename(columns=column_renaming)

# --- Aggregated columns ---
def is_valid_date(year, month, day):
    """Check if the given year, month, day form a valid date."""
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False

def calculate_duration(row):
    if not is_valid_date(row['Year'], row['Month'], row['Day']):
        return -99
    
    if row['Extended'] == 1 and pd.notnull(row['Resolution']):
        attack_date = datetime(row['Year'], row['Month'], row['Day'])
        try:
            resolution_date = datetime.strptime(row['Resolution'], "%m/%d/%Y")
            return (resolution_date - attack_date).days
        except ValueError:
            return 1
    return 1

filtered_df['Duration'] = filtered_df.apply(calculate_duration, axis=1)
filtered_df = filtered_df.drop(columns=['Extended', 'Resolution'])

def calculate_casualties(row):
    if pd.isnull(row['Number of Killed People']) or pd.isnull(row['Number of Wounded People']):
        return -99
    return row['Number of Killed People'] + row['Number of Wounded People']

filtered_df['Number of Casualties'] = filtered_df.apply(calculate_casualties, axis=1)

# Replace null values with -99
filtered_df['Number of Terrorists'] = filtered_df['Number of Terrorists'].fillna(-99)
filtered_df['Number of Killed People'] = filtered_df['Number of Killed People'].fillna(-99)
filtered_df['Number of Wounded People'] = filtered_df['Number of Wounded People'].fillna(-99)
filtered_df['Number of Killed US People'] = filtered_df['Number of Killed US People'].fillna(-99)
filtered_df['Number of Wounded US People'] = filtered_df['Number of Wounded US People'].fillna(-99)
filtered_df['Target Nationality'] = filtered_df['Target Nationality'].fillna('Unknown')

# --- Sample Selection ---
filtered_df['Decade'] = (filtered_df['Year'] // 10) * 10

sampling_fraction = 0.1

# Sample 10% of each group by Decade and Region
sampled_df = filtered_df.groupby(['Decade', 'Region']).apply(lambda x: x.sample(frac=sampling_fraction, random_state=1)).reset_index(drop=True)

# --- Integration ---
#  Merge GDP Data
gdp_df = pd.read_csv('Preprocessed_GDP_Dataset.csv')

merged_df = pd.merge(filtered_df, gdp_df, how='left', left_on=['Year', 'Country'], right_on=['Year', 'Country'])

merged_df = merged_df.drop(columns=['Country'])

output_file_path = 'Preprocessed_Global_Terrorism_Dataset.csv'
merged_df.to_csv(output_file_path, index=False)

print(f'Data saved to {output_file_path} with selected columns, new names, and GDP information.')

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
output_file_path = 'PCA_Analysis.csv'
reduced_df.to_csv(output_file_path, index=False)

# --- Detecting Anomalies ---
# ---- Contextual Anomalies ----
kla_rows = filtered_df[filtered_df['Attacking Group Name'].str.contains("KLA", na=False)]
removed_count = kla_rows.shape[0]
filtered_df = filtered_df[~filtered_df['Attacking Group Name'].str.contains("KLA", na=False)]
print(f"\n\n-------------------------------------------------------------------------------\n")
print(f"Detecting Anomalies \n\n")

print(f"Detecting and handling Contextual Anomalies")
print(f"\n{removed_count} rows removed where 'KLA' is considered as terrorist organisation.\n\n")

# ---- Proximity-Based outlier detection ----
file_path = "Preprocessed_Global_Terrorism_Dataset.csv" 
df = pd.read_csv(file_path)

print("\nProximity-based outlier detection for 'Number of Killed US People', 'Number of Wounded US People'")
feature_columns = ['Number of Killed US People', 'Number of Wounded US People']

df_filtered = df[feature_columns].replace(-99, None).dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_filtered)

def knn_anomaly_detection(k, scaled_data, original_data):
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(scaled_data)
    
    distances, indices = nbrs.kneighbors(scaled_data)
    kth_distances = distances[:, -1]
    
    original_data['kth_distance'] = kth_distances
    
    top_anomalies = original_data.sort_values('kth_distance', ascending=False).head(5)
    
    print(f"\nTop 5 anomalies with k = {k}:")
    print(top_anomalies[['kth_distance'] + feature_columns])

for k in [1, 20]:
    knn_anomaly_detection(k, scaled_features, df_filtered.copy())

def statistical_anomaly_detection_z_score(data, column, threshold=3):
    valid_data = data[data[column] != -99].copy()
    
    mean = valid_data[column].mean()
    std = valid_data[column].std()
    
    valid_data['Z Score'] = (valid_data[column] - mean) / std
    anomalies = valid_data[np.abs(valid_data['Z Score']) > threshold]
    
    anomaly_percentage = (len(anomalies) / len(valid_data)) * 100 if len(valid_data) > 0 else 0
    print(f"\nStatistical-Based Anomalies using Z-Score for {column}: {anomaly_percentage:.2f}%")
    
    if not anomalies.empty:
        top_5_anomalies = anomalies.assign(abs_score=anomalies['Z Score'].abs()).sort_values('abs_score', ascending=False).head(5)
        print("\nTop 5 anomalies by highest absolute z-score:")
        print(top_5_anomalies[['Year', 'Country', column, 'Z Score']])
    else:
        print("No anomalies found.")
    
    return anomalies
stat_anomalies_killed = statistical_anomaly_detection_z_score(filtered_df, 'Number of Killed People')
stat_anomalies_wounded = statistical_anomaly_detection_z_score(filtered_df, 'Duration')

def detect_highest_anomaly_grubbs(data, column, significance_level=0.05):
    cleaned_data = data[data[column] != -99].copy()
    print("\nStatistical-Based Anomalies using Grubb's Test for Number of Wounded People")

    mean = cleaned_data[column].mean()
    std_dev = cleaned_data[column].std()
    n = len(cleaned_data)

    G = abs(cleaned_data[column].max() - mean) / std_dev
    t_critical = t.ppf(1 - significance_level / (2 * n), n - 2)
    critical_value = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))

    print(f"n = {n}, α = {significance_level}, Gmax = {G:.4f}, G critical = {critical_value:.4f}")

    if G > critical_value:
        anomaly_value = cleaned_data[column].max()
        anomaly_row = data[data[column] == anomaly_value].iloc[0]
        print("\nHighest Anomaly Detected:")
        print(f"Detected anomaly: {anomaly_value}")
        return anomaly_row
    else:
        print("No anomaly detected.")
        return None


highest_anomaly = detect_highest_anomaly_grubbs(filtered_df, 'Number of Wounded People')

if highest_anomaly is not None:
    print("\nSummary of the highest detected anomaly:")
    print(f"Year: {highest_anomaly['Year']}, Region: {highest_anomaly['Region']}, "
          f"Number of Wounded People: {highest_anomaly['Number of Wounded People']}")
else:
    print("\nNo anomalies detected.")

# --- Density Based Anomaly Detection ---
def density_based_anomaly_detection(data, columns, n_neighbors=20):
    valid_data = data[(data[columns] != -99).all(axis=1)][columns].dropna().copy()
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean')
    valid_data['density_score'] = lof.fit_predict(valid_data)
    valid_data['Anomaly Score'] = -lof.negative_outlier_factor_

    anomalies = valid_data[valid_data['density_score'] == -1].copy()

    anomalies['Duration'] = data.loc[anomalies.index, 'Duration']
    anomalies['Year'] = data.loc[anomalies.index, 'Year']
    anomalies['Country'] = data.loc[anomalies.index, 'Country']

    top_5_anomalies = anomalies.sort_values(by='Anomaly Score', ascending=False).head(5)
    anomaly_percentage = (len(anomalies) / len(valid_data)) * 100
    print(f"\nDensity-Based Anomalies for {columns} {anomaly_percentage:.2f}%")

    print("\nTop 5 Anomalies:")
    print(top_5_anomalies[['Year', 'Country', 'Number of Killed People','Duration', 'Anomaly Score']])

    return anomalies

density_combo = ['Number of Killed People', 'Duration']
density_anomalies = density_based_anomaly_detection(filtered_df, density_combo)

# --- Clustering Based Anomaly Detection ---
def clustering_based_anomaly_detection(data, columns, eps=0.8, min_samples=10):
    for col in ['Year', 'Country', 'Weapon Type', 'Attack Type']:
        if col not in data.columns:
            data[col] = '-'
    
    available_columns = [col for col in columns if col in data.columns]
    if not available_columns:
        print("No valid columns available for clustering.")
        return None

    
    valid_data = pd.get_dummies(data[available_columns].dropna())
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    valid_data['cluster'] = dbscan.fit_predict(valid_data)
    
    anomalies = valid_data[valid_data['cluster'] == -1].copy()
    
    for col in ['Year', 'Country', 'Weapon Type', 'Attack Type']:
        anomalies[col] = data.loc[anomalies.index, col]
    
    anomalies['Anomaly Score'] = 1 
    top_5_anomalies = anomalies.sort_values(by='Anomaly Score', ascending=False).head(5)
    
    anomaly_percentage = (len(anomalies) / len(valid_data)) * 100
    print(f"\nClustering-Based Anomalies for {available_columns}: {anomaly_percentage:.2f}%")

    print("\nTop 5 Anomalies:")
    columns_to_print = ['Year', 'Country', 'Weapon Type', 'Attack Type', 'Anomaly Score']
    print(top_5_anomalies[columns_to_print])
    
    return anomalies

clustering_combo = ['Weapon Type', 'Attack Type']
cluster_anomalies = clustering_based_anomaly_detection(filtered_df, clustering_combo)

required_columns = ['Anomaly Score', 'Z Score', 'Detection Method', 'Year', 'Country', 'Number of Killed People', 'Duration', 'Weapon Type', 'Attack Type']
default_value = '-'

for anomalies_df, detection_method in [
    (stat_anomalies_killed, 'Statistical Z-Score (Killed)'),
    (stat_anomalies_wounded, 'Statistical Z-Score (Wounded)'),
    (density_anomalies, 'Density-Based'),
    (cluster_anomalies, 'Clustering-Based')
]:
    for col in required_columns:
        if col not in anomalies_df.columns:
            anomalies_df[col] = default_value 
    anomalies_df['Detection Method'] = detection_method

all_anomalies_combined = pd.concat(
    [
        stat_anomalies_killed[required_columns],
        stat_anomalies_wounded[required_columns],
        density_anomalies[required_columns],
        cluster_anomalies[required_columns]
    ],
    axis=0
).reset_index(drop=True)

output_csv_path = 'Detected_Anomalies.csv'
all_anomalies_combined.to_csv(output_csv_path, index=False)
print(f"Anomalies saved to {output_csv_path}")

# --- Clean out data ---
filtered_df_cleaned = filtered_df[
    ((filtered_df['Duration'] >= 1) | (filtered_df['Duration'] == -99)) & (filtered_df['Duration'] != 7324)
]
filtered_df_cleaned = filtered_df_cleaned[~(
    ((filtered_df_cleaned['Weapon Type'] == 'Chemical') & (filtered_df_cleaned['Attack Type'] == 'Armed Assault')) |
    ((filtered_df_cleaned['Weapon Type'] == 'Explosives') & (filtered_df_cleaned['Attack Type'] == 'Unarmed Assault')) |
    ((filtered_df_cleaned['Weapon Type'] == 'Melee') & (filtered_df_cleaned['Attack Type'] == 'Bombing/Explosion')) |
    ((filtered_df_cleaned['Weapon Type'] == 'Fake Weapons') & (filtered_df_cleaned['Attack Type'] == 'Bombing/Explosion')) |
    ((filtered_df_cleaned['Weapon Type'] == 'Fake Weapons') & (filtered_df_cleaned['Attack Type'] == 'Facility/Infrastructure'))
)]

cleaned_output_path = 'Filtered_Global_Terrorism_Dataset.csv'
filtered_df_cleaned.to_csv(cleaned_output_path, index=False)

print(f"Filtered dataset saved to {cleaned_output_path}")
