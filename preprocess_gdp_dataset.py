import pandas as pd
import zipfile

# --- Uniqueness ---
def get_unique_terrorism_countries(zip_file_path, file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(file_name) as file:
            terrorism_df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
    return set(terrorism_df['country_txt'].unique())

unique_terrorism_countries = get_unique_terrorism_countries('GlobalTerrorismDataset.zip', 'globalterrorismdb_0718dist.csv')

country_name_mapping = {
    "Bahamas, The": "Bahamas", "Bosnia and Herzegovina": "Bosnia-Herzegovina", "Brunei Darussalam": "Brunei", 
    "Congo, Rep.": "Republic of the Congo", "Czechia": "Czech Republic", "Egypt, Arab Rep.": "Egypt", 
    "Gambia, The": "Gambia", "Hong Kong SAR, China": "Hong Kong", "Iran, Islamic Rep.": "Iran", 
    "Cote d'Ivoire": "Ivory Coast", "Kazakhstan": "Kazakhstan", "Kyrgyz Republic": "Kyrgyzstan", 
    "Lao PDR": "Laos", "Macao SAR, China": "Macau", "North Macedonia": "Macedonia", 
    "Korea, Dem. People's Rep.": "North Korea", "Russian Federation": "Russia", "Korea": "South Korea", 
    "Eswatini": "Swaziland", "Syrian Arab Republic": "Syria", "Taiwan, Province of China": "Taiwan", 
    "Turkiye": "Turkey", "Yemen, Rep.": "South Yemen", "Congo, Dem. Rep.": "People's Republic of the Congo", 
    "Former Yugoslavia": "Yugoslavia", "Timor-Leste": "East Timor", "Former USSR": "Soviet Union", 
    "Zimbabwe": "Rhodesia", "Viet Nam": "South Vietnam", "Vanuatu": "New Hebrides", "Venezuela, RB": "Venezuela"
}

def process_maddison_project(zip_file_path, file_name, sheet_name='GDPpc'):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(file_name) as file:
            df = pd.read_excel(file, sheet_name=sheet_name)
    
    countries = ['Taiwan, Province of China', 'Former Yugoslavia', 'Czechoslovakia', 'Former USSR']
    df['GDP pc 2011 prices'] = pd.to_numeric(df['GDP pc 2011 prices'], errors='coerce')
    df_filtered = df[['GDP pc 2011 prices'] + countries]
    df_filtered = df_filtered.dropna(subset=['GDP pc 2011 prices'])
    df_filtered = df_filtered[df_filtered['GDP pc 2011 prices'].between(1970, 2017)]
    df_filtered = df_filtered.rename(columns={'GDP pc 2011 prices': 'Year'})
    df_melted = df_filtered.melt(id_vars=["Year"], var_name="Country Name", value_name="GDP")
    return df_melted

def process_world_bank(zip_file_path, file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(file_name) as file:
            df = pd.read_csv(file, skiprows=4, low_memory=False)
    
    df = df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
    years = [str(year) for year in range(1970, 2018)]
    df = df[['Country Name'] + years]
    df_melted = df.melt(id_vars=["Country Name"], var_name="Year", value_name="GDP")
    return df_melted

# --- Integration ---
maddison_data = process_maddison_project('GDP_Maddison_Project_Database.zip', 'GDP_Maddison_Project_Database.xlsx')
world_bank_data = process_world_bank('GDP_World_Bank_Group.zip', 'GDP_World_Bank_Group.csv')

combined_data = pd.concat([maddison_data, world_bank_data])

combined_data['Country Name'] = combined_data['Country Name'].replace(country_name_mapping)

filtered_data = combined_data[combined_data['Country Name'].isin(unique_terrorism_countries)]

additional_mappings = {
    "Zaire": "Congo, Dem. Rep.", "Serbia-Montenegro": "Serbia", "West Germany (FRG)": "Germany", 
    "East Germany (GDR)": "Germany", "Vatican City": "Italy", "Wallis and Futuna": "France", 
    "French Guiana": "France", "Martinique": "France", "Guadeloupe": "France"
}

for new_country, ref_country in additional_mappings.items():
    new_country_data = filtered_data[filtered_data['Country Name'] == ref_country].copy()
    new_country_data['Country Name'] = new_country
    filtered_data = pd.concat([filtered_data, new_country_data])

countries_with_blank_gdp = ["Western Sahara", "International", "South Sudan", "Falkland Islands"]
for country in countries_with_blank_gdp:
    blank_data = pd.DataFrame({
        "Year": range(1970, 2018),
        "Country Name": country,
        "GDP": [None] * (2018 - 1970)
    })
    filtered_data = pd.concat([filtered_data, blank_data])

filtered_data.loc[:, 'Year'] = pd.to_numeric(filtered_data['Year'], errors='coerce')

# --- Handling Missing Values ---
filtered_data['GDP'] = filtered_data['GDP'].fillna(-99)

# --- Rename and Sort ---
filtered_data_sorted = filtered_data.rename(columns={"Country Name": "Country"}).sort_values(by=["Country", "Year"])

filtered_data_sorted.to_csv('Preprocessed_GDP_Dataset.csv', index=False)
