# Global Terrorism Dataset Analysis 📊

This project focuses on analyzing the **Global Terrorism Dataset** to uncover patterns, trends, and insights into global terrorist incidents. The dataset, sourced from **Kaggle**, contains detailed information on over **180,000** attacks worldwide from **1970 to 2017**. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism **(START)**, headquartered at the University of Maryland.

You can find the dataset here: [Global Terrorism Dataset on Kaggle](https://www.kaggle.com/datasets/START-UMD/gtd).

The project is structured in three phases **Data Pre-Processing**, **Outlier Detection and Removal** dhe **Data Visualization**.

## I. Data Pre-Processing 🛠️

- **Data Collection:** Accessing the Global Terrorism Dataset on Kaggle and analysing different db sources of dataset
- **Data Integration:** Integrading the GPD dataset with the GTD dataset
- **Data Type Definition:** Nominal, Ordinal, Interval, Ratio
- **Data Quality:** Completeness, Uniqueness, Accuracy Checks, Check for missing values
- **Integration, Aggregation, and Display:** Agregating columns such as Duration and Number of Casualties
- **Data Cleaning:** On different columns such as: Number of Terrorists, Number of Killed People, Number of Wounded People etc.
- **Dimension Reduction:** Employed Principal Component Analysis (PCA) on selected features to reduce the dataset's dimensionality.
- **Subset Selection:** Created a decade-based sample selection, sampling 10% of each group by decade and region.
- **Discretization and Binarization:** Grouped casualty counts into severity ranges for trend analysis; converted key variables into binary values to simplify categorical comparisons.
___

<table align="center"><tr><td align="center"><strong>Data Collection</strong></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/975ecc1b-5daa-455d-91ec-60f00e5c9ad9" alt="Data Collection"></td></tr></table>

<table align="center"><tr><td align="center"><strong>Data Quality</strong></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/8f988be9-50ca-42ec-99d4-5eb3c8d1620c" alt="Completeness Analysis"></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/14a2c5af-1f80-49c3-a99e-096389ddbdaf" alt="Completeness Analysis"></td></tr</table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/213ba1cc-bff2-4133-a758-883d47ccb6f2" alt="Uniqueness Analysis"></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/db7e95f4-a303-48e2-87ec-580a0bda4b99" alt="Uniqueness Analysis"></td></tr></table>

<table align="center"><tr><td align="center"><strong>Selection Of The Subset Of Attributes</strong></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/bea317f2-3e53-4178-9e24-dd81120dc5c5" alt="Selection Of The Subset Of Attributes"></td></tr></table>

<table align="center"><tr><td align="center"><strong>Type of Attributes Classification</strong></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/e393b8fe-9a5d-4c93-a5b7-d64c25b2947b" alt="Type of Attributes Classification"></td></tr></table>

<table align="center"><tr><td align="center"><strong>Binning</strong></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/04aab7a2-59a1-4fc9-b4f6-17caf8793891" alt="Binning"></td></tr></table>

<table align="center"><tr><td align="center"><strong>Decade Distribution</strong></td></tr></table>

<table align="center"><tr><td><img src="https://github.com/user-attachments/assets/90d6ab3c-6c98-45e9-8cfb-13da3d296d18" alt="Decade Distribution"></td></tr></table>

___
🏷️ **License**: This project is open to use for anyone. You are free to use, modify, and distribute the code as needed.
