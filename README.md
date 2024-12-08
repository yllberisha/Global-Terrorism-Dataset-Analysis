# Global Terrorism Dataset Analysis 📊

<table>
  <tr>
   <td>
     <img src="https://github.com/user-attachments/assets/97d82b15-7058-4498-bdd4-efbae2425810" alt="University Logo" width="200" >
    </td>
    <td>
      <h2>UNIVERSITETI I PRISHTINËS “HASAN PRISHTINA”</h2>
      <p><strong>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p><strong>Departamenti: </strong> Inxhinieri Kompjuterike dhe Softuerike</p>
      <p><strong>Kodra e Diellit, p.n. - 10000 Prishtinë, Kosova</string>
    </td>
   
  </tr>
</table>

**Level of Studies:** *Master*   <br>
**Students:** *[Jetë Lajçi](https://github.com/Jeta52), [Melisa Alaj](https://github.com/melisaalaj), [Yll Berisha](https://github.com/yllberisha)*   <br>
**Professor:** *[Dr. Sc. Mërgim H. HOTI](https://github.com/MergimHHoti)*   <br>
**------------------------------------------------------------------------------------------------------------------------------------------------------------**


This project focuses on analyzing the **Global Terrorism Dataset** to uncover patterns, trends, and insights into global terrorist incidents. The dataset, sourced from **Kaggle**, contains detailed information on over **180,000** attacks worldwide from **1970 to 2017**. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism **(START)**, headquartered at the University of Maryland.

You can find the dataset here: [Global Terrorism Dataset on Kaggle](https://www.kaggle.com/datasets/START-UMD/gtd).

The project is structured in three phases **Data Pre-Processing**, **Outlier Detection and Removal** dhe **Data Visualization**.

## I. Data Pre-Processing 🛠️
* **`Data Collection:`** The Global Terrorism Dataset has a column called **Database Source** that shows where the information for each incident came from.
  <table align="center"><tr><td><img src="https://github.com/user-attachments/assets/975ecc1b-5daa-455d-91ec-60f00e5c9ad9" alt="Data Collection"></td></tr></table>
* **`Data Integration:`** The Global Terrorism Dataset does not include a **GDP** column, so we integrated it with two other datasets: **GDP_Maddison_Project_Database** and **GDP_World_Bank_Group**, to add GDP data.
* **`Data Quality:`**
   * **`Completeness:`** The dataset contains attributes with missing data. Below is a table showing the **Missing Value Count** and **Completeness Percentage** for each attribute.
   <p align="center">
      <img src="https://github.com/user-attachments/assets/8f988be9-50ca-42ec-99d4-5eb3c8d1620c" alt="image" width="600px">
    </p>

   <p align="center">
      <img src="https://github.com/user-attachments/assets/14a2c5af-1f80-49c3-a99e-096389ddbdaf" alt="image" width="600px">
    </p>

   * **`Uniqueness:`** This analysis evaluates the **distinct** values for each attribute in the dataset. The table lists the attributes along with their respective count of unique values

   <p align="center">
      <img src="https://github.com/user-attachments/assets/213ba1cc-bff2-4133-a758-883d47ccb6f2" alt="image" width="600px">
    </p>
   
   * **`Duplicate values:`** This analysis checks for repeated rows in the dataset. The results show that **no duplicate** rows were found, indicating that all rows in the dataset are unique.
   <p align="center">
      <img src="https://github.com/user-attachments/assets/db7e95f4-a303-48e2-87ec-580a0bda4b99" alt="image" width="600px">
    </p>

* **`Aggregation:`** Agregated columns such as **Duration** (Extended and Resolution) and **Number of Casualties** (Number of Killed People and Number of Wounded People)
* **`Data Cleaning:`** On different columns such as: **Number of Terrorists, Number of Killed People, Number of Wounded People** etc.
* **`Dimension Reduction:`** Employed Principal Component Analysis (PCA) on selected features to reduce the dataset's dimensionality.
* **`Subset Selection:`** Created a **decade-based sample selection**, sampling **10%** of each group by decade and region.
  <table align="center"><tr><td><img src="https://github.com/user-attachments/assets/bea317f2-3e53-4178-9e24-dd81120dc5c5" alt="Selection Of The Subset Of Attributes"></td></tr></table>
  <table align="center"><tr><td><img src="https://github.com/user-attachments/assets/e393b8fe-9a5d-4c93-a5b7-d64c25b2947b" alt="Type of Attributes Classification"></td></tr></table>

* **`Binning:`** This analysis groups data into predefined ranges, such as the number of victims per range. The table shows the **victim ranges** (e.g., 0-10, 11-100) along with the count of occurrences in each range.

   <p align="center">
      <img src="https://github.com/user-attachments/assets/04aab7a2-59a1-4fc9-b4f6-17caf8793891" alt="image" width="200px">
    </p>

* **`Decade Distribution:`** This analysis categorizes data based on the decade in which events occurred. The table lists the **decades** (e.g., 2010s, 1990s) along with the number of events for each.

   <p align="center">
      <img src="https://github.com/user-attachments/assets/90d6ab3c-6c98-45e9-8cfb-13da3d296d18" alt="image" width="200px">
    </p>

# II. Outlier Detection and Removal 🔍
**Anomalies** and **outliers** are essentially the same thing: objects that are **different** from most other objects.
Said differently, an outlier is an observation so different from the others that one suspects that it was generated by a different mechanism.

An anomaly detection system can uncover two general types of anomalies: **unintentional** and **intentional**. 
* **Unintentional anomalies** are deviations caused by errors or noise, such as faulty sensors or data entry mistakes, which can mess up the data and make it harder to analyze correctly.
* **Intentional anomalies** are deviations caused by real-world events or actions, providing valuable insights into unique trends or occurrences. [[2]](https://www.ibm.com/topics/anomaly-detection#:~:text=Unintentional%20anomalies%20can%20distort%20the,highlight%20unique%20occurrences%20or%20trends.)

## Structure of anomalies
* **`Point anomalies`** - An individual data instance is anomalous with respect to the data.<br><br>
*The **9/11** attacks had **8,190 wounded people** (Number of Wounded People), making it a significant point anomaly. This is an intentional point anomaly because it was caused by a real-world event.*

     <p align="center">
      <img src="https://github.com/user-attachments/assets/7c76be4c-7715-4501-ae64-d7b2eebc2576" alt="image" width="600px">
    </p>

* **`Contextual anomalies`** - An individual data instance is anomalous within a context.<br><br>
*There are **94** rows in the dataset where the **KLA (Kosovo Liberation Army)** is considered a terrorist organization. This can be classified as a contextual anomaly because it reflects a biased perspective. The labeling of the KLA as a terrorist organization is a completely wrongful classification, and this misclassification could lead to incorrect or biased conclusions.*

     <p align="center">
      <img src="https://github.com/user-attachments/assets/2779cc81-253e-49ce-8663-1ff210deedbf" alt="image" width="600px">
    </p>

* **`Collective anomalies`** - A group of data instances that are anomalous when considered together, even if the individual data points within the group might not appear anomalous on their own.<br><br>
*All incidents from **1993** are missing because the original data was lost prior to the database's compilation. Efforts to recover the data identified only 15% of the estimated cases, leading to significant gaps.
As a result, 1993 is excluded from analysis to avoid misinterpreting the low frequency of incidents as actual data.*

     <p align="center">
      <img src="https://github.com/user-attachments/assets/c2906cf4-4541-4a0b-93ba-1c531fd2d44a" alt="image" width="600px">
    </p>

## Techniques for anomaly detection
* **`Statistical`** 
  * **Calculation of z-score** - The z-score tells us how far a specific **number (data point)** is from the **average (mean)** of a dataset. It uses **standard deviations** to measure this distance. The **standard deviation** is a number that shows how spread out the data is around the mean:
    * If the data points are close to the mean, the standard deviation is **small**.
    * If the data points are spread far from the mean, the standard deviation is **large**.
      
    <p align="center">
      <img src="https://github.com/user-attachments/assets/ef98f1c7-cac7-4d1f-bb95-07527c440ad1" alt="image" width="200px">
    </p>

     <p align="center">
      <img src="https://github.com/user-attachments/assets/8c4d6987-6156-47b3-a50a-abbcc757617b" alt="image" width="700px">
    </p>
    
After calculating the z-score, we often refer to a [z-score table](https://z-table.com/) to determine the probability or percentage of values that fall below or above a specific z-score.
<br>
    
  * **Grubbs’ test** - It evaluates whether the largest absolute deviation from the mean (or smallest, depending on the tail) is significantly different from other data points. For the Maximum Value:

     <p align="center">
      <img src="https://github.com/user-attachments/assets/a04a652b-f741-46ca-8804-705b4c4e54cc" alt="image" width="200px">
    </p>

     <p align="center">
      <img src="https://github.com/user-attachments/assets/ad486aaf-6199-4bba-8332-ea4e858f981c" alt="image" width="500px">
    </p>

    * Compare **𝐺max** to a critical value (from a [Grubbs distribution table](http://www.statistics4u.com/fundstat_eng/ee_grubbs_outliertest.html)).
    * If **𝐺max** is larger than the critical value, then  **𝑋max** is considered an outlier.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/f515e19d-9dec-4cb3-80a7-eae4d453f5fe" alt="image" width="700px">
    </p>

* **`Proximity-based`** - Identifies anomalies by measuring the **distance between data points** in a dataset. It assumes that normal data points are **close** to each other, while anomalies are **far** away from their nearest neighbors. (Distance-Based Outlier Detection)
    * **k = 1**  The algorithm only considers the distance to the closest neighbor.
    * **k = 20** The algorithm considers the average distance to the 20 nearest neighbors.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/5dd1eaab-b500-4b9f-a38f-b268ba2ad81d" alt="image" width="700px">
    </p>

* **`Density-based`** - **Outliers** are objects in regions of **low density**.
    <p align="center">
      <img src="https://github.com/user-attachments/assets/c98f8d76-2828-4ed7-bc74-f0748454fd94" alt="image" width="700px">
    </p>
    
* **`Clustering-based`** - **Outliers** are objects that **do not** belong strongly to any **cluster**.
    <p align="center">
      <img src="https://github.com/user-attachments/assets/aed21f1e-9951-4c21-905b-0d40484cbd5d" alt="image" width="700px">
    </p>

## Similarity and Dissimilarity

* **Similarity**: Measures how much two data points are alike. Usually represented as a number between **0** and **1** (e.g., 1 means identical, 0 means completely different).
* **Dissimilarity**: Measures how different two data points are. Starts at **0** (identical points) and can go higher, depending on the scale.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/5d9de2c5-b43b-4ed6-b4f8-a543a7255945" alt="image" width="700px">
    </p>
  ___
🏷️ **License**: This project is open to use for anyone. You are free to use, modify, and distribute the code as needed.
