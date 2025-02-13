# Exploratory Data Analysis & Feature Engineering on Travel Dataset

## 📌 Project Overview  
This project focuses on performing **Exploratory Data Analysis (EDA)** and **Feature Engineering** on the **Travel** dataset.  
The goal is to **understand the data, identify patterns, handle missing values, detect outliers, and create meaningful features** to enhance predictive modeling.  

### 🔹 Key Objectives:  
- **Data Ingestion** – Loading and understanding the dataset.  
- **Data Cleaning** – Handling missing values, duplicates, and inconsistencies.  
- **Exploratory Data Analysis (EDA)** – Uncovering insights through visualizations.  
- **Feature Engineering** – Creating new features to improve model performance.  

By the end of this project, the dataset will be transformed into a **cleaned and enriched version** ready for **further modeling or analysis**.  

## 📊 Dataset Information  
The dataset used in this project is **Travel Dataset**, which contains customer-related information for a travel company. The dataset includes details about customer demographics, preferences, interactions, and responses to travel offers.  

### 🔹 Dataset Source:  
- **Source:** Received from my institute, where I am learning Data Analytics and Data Science.  
- **Description:** This dataset is provided as part of my coursework to practice **Exploratory Data Analysis (EDA) and Feature Engineering**.  

### 🔹 Key Features:  
| Column Name                | Description |
|----------------------------|-------------|
| `CustomerID`               | Unique identifier for each customer |
| `ProdTaken`                | Whether the customer purchased the travel package (1 = Yes, 0 = No) |
| `Age`                      | Age of the customer |
| `TypeofContact`            | How the customer was contacted (e.g., Self-Initiated, Company-Initiated) |
| `CityTier`                 | Tier of the city the customer belongs to (1, 2, or 3) |
| `DurationOfPitch`          | Duration of the sales pitch (in minutes) |
| `Occupation`               | Profession of the customer |
| `Gender`                   | Gender of the customer |
| `NumberOfPersonVisiting`   | Number of people accompanying the customer on the trip |
| `NumberOfFollowups`        | Number of follow-ups made to the customer |
| `ProductPitched`           | Type of travel package pitched to the customer |
| `PreferredPropertyStar`     | Customer’s preferred hotel star rating (e.g., 3-star, 4-star) |
| `MaritalStatus`            | Marital status of the customer (e.g., Married, Single) |
| `NumberOfTrips`            | Number of trips the customer has taken |
| `Passport`                 | Whether the customer has a passport (1 = Yes, 0 = No) |
| `PitchSatisfactionScore`   | Satisfaction score given by the customer for the sales pitch |
| `OwnCar`                   | Whether the customer owns a car (1 = Yes, 0 = No) |
| `NumberOfChildrenVisiting` | Number of children accompanying the customer |
| `Designation`              | Job title of the customer |
| `MonthlyIncome`            | Monthly income of the customer |

### 🔹 Dataset Size:  
- **Number of rows:** 4,888  
- **Number of columns:** 20  
- **Missing values:** 1,012  
- **Duplicate rows:** 0  

 ## 🔧 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)
- Google Collab

 
## 🔄 Data Ingestion & Preprocessing  

Before performing Exploratory Data Analysis (EDA), the dataset was **cleaned and preprocessed** to handle inconsistencies and missing values. This step ensures that our dataset is structured, reliable, and ready for further analysis.  

### 🔹 1. Data Loading & Initial Exploration  
The dataset was loaded using **pandas**, and basic exploratory commands were used to understand its structure:  

```python
import pandas as pd  
df = pd.read_csv("Travel.csv")  
df.shape  # Check dataset dimensions  
df.head()  # View first five rows  
df.info()  # Check data types and missing values  
df.dtypes  # Identify column data types  
df.columns  # List all column names  


Observations:

The dataset contains 4888 rows and 10+ columns.
Some categorical features have inconsistent values (e.g., "Fe Male" instead of "Female").
Missing values exist in some columns, requiring handling.

```
### 🔹 2. Handling Missing Values
```python
df.isnull().sum().sort_values(ascending=False)
```
Findings:
Columns like TypeofContact and MonthlyIncome had significant missing values.
Instead of using imputation, rows with missing values were dropped, since they were a small percentage of the dataset.

 ```python
df.dropna(axis=0, inplace=True)  # Remove rows with missing values  
df.isnull().sum()  # Verify missing values are handled
```
Post-cleaning, the dataset had 4000+ rows, ensuring a clean dataset for analysis.

###🔹 3. Fixing Data Inconsistencies

Upon checking categorical values, inconsistencies were found in Gender, where "Fe Male" was incorrectly recorded. This was corrected as follows:

```python
df.replace("Fe Male", "Female", inplace=True)  
df.Gender.unique()  # Verify changes
Other categorical inconsistencies were checked and handled similarly.
```

### 🔹 4. Identifying Categorical & Numerical Features  

After handling missing values and fixing inconsistencies, the next step was to categorize the features into **categorical** and **numerical** groups.  

Since **all columns were initially stored as numeric data types**, some categorical features were mapped using numerical values. These columns needed to be identified and treated accordingly for accurate analysis.  

To separate them:  

```python
categorical_features = [
    "TypeofContact", "Occupation", "MaritalStatus", 
    "Gender", "ProductPitched", "Designation"
]

numerical_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting", 
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome"
]
```

## 🔄 Exploratory Data Analysis (EDA)

### 🔹 1. Univariate Analysis  

Univariate analysis focuses on examining each feature **individually** to understand its distribution and unique characteristics.  

#### 🔸 Categorical Features Distribution  

To analyze categorical features, **count plots** were used to visualize the frequency of each category. This helps in identifying **dominant categories, class imbalances, and potential insights**.  

For example, the distribution of the **target variable** `ProdTaken` (whether a customer purchased the product or not) was analyzed:  

```python
plt.figure(figsize=(8,5))
sns.countplot(x='ProdTaken', data=df, palette='plasma')
plt.show()
```
![download](https://github.com/user-attachments/assets/c10bf906-cfc5-41e8-91b5-f54a86c48e91)

 
 
To efficiently analyze the distribution of categorical columns, I plotted **count plots** for each feature. This helped me understand **category frequencies, potential class imbalances, and customer preferences**.  

Instead of manually plotting each feature, I used a **for loop** to automate the process:  

```python

for cat_column in cats:
  plt.figure(figsize = (8,5))
  sns.countplot(x = cat_column, data = df, palette = 'viridis')
  plt.title(f'Countplot(Univariate Analysis) for {cat_column}')
  plt.show()
```

By plotting all categorical columns at once, I was able to get a comprehensive view of data distribution, making it easier to spot patterns and irregularities.

![download](https://github.com/user-attachments/assets/b390d65d-2dab-4543-ae10-6cdd06e60c61)

The ProdTaken countplot shows that the majority of customers (0) did not purchase the product, with only a smaller proportion (1) making a purchase. This indicates a low conversion rate, suggesting potential issues in marketing, pricing, or customer interest.

![download](https://github.com/user-attachments/assets/d9d5f37b-b43e-4aba-9e6a-c52ebecf6c0e)

The **CityTier** distribution shows that most customers belong to **Tier 1 cities**, followed by **Tier 3**, while **Tier 2 cities** have the least representation. This suggests that the business is primarily targeting or attracting customers from **highly developed urban areas**.

![download](https://github.com/user-attachments/assets/2935d9bb-a3c0-4f7a-85a1-11965aa42156)


Most customers prefer 3-star properties, while 4-star and 5-star properties have a significantly lower but nearly equal preference. This suggests that affordability and mid-range accommodations are more popular among the target audience.


![download](https://github.com/user-attachments/assets/1dbb551e-d06c-4519-ae9e-0c0c256f2011)

A majority of customers do not have a passport, which may indicate a stronger focus on domestic travel rather than international travel.

![download](https://github.com/user-attachments/assets/09ab5ecb-6106-4eaa-943f-c5841591e551)


A larger proportion of customers own a car, which could indicate a preference for road trips or self-driven travel rather than relying on public transportation or flights.

![download](https://github.com/user-attachments/assets/be83f108-c2d2-43fa-b327-e5245aadaf40)


The most common Pitch Satisfaction Score is 3, indicating that a majority of customers have a neutral opinion about their experience. However, there is also a significant proportion of customers who rated it as 1 or 5, suggesting a mix of highly dissatisfied and highly satisfied customers.


![download](https://github.com/user-attachments/assets/aebd3f6e-ab10-4d0d-a191-84225313a657)


The majority of contacts are from Self Enquiry, meaning most customers are reaching out on their own rather than being invited by the company. This could indicate a strong interest from customers but might also mean that proactive outreach by the company is limited.

![download](https://github.com/user-attachments/assets/cc261b08-3f99-426d-9ca9-cc3247d5bcdc)


The majority of customers are salaried employees and small business owners, while freelancers and large business owners are significantly fewer. This suggests that the product/service is more appealing to working professionals and small entrepreneurs rather than large businesses or independent freelancers.

![download](https://github.com/user-attachments/assets/9ffe89aa-ae2d-4cea-86b2-3c5b54ebbefe)

More male customers than female.

![download](https://github.com/user-attachments/assets/000ad5dd-d483-49fc-8e96-fe3650b569f7)

Basic & Deluxe are most pitched, while Super Deluxe & King are least.

![download](https://github.com/user-attachments/assets/864ce09f-69ba-4470-883e-c448cc0c2eb7)

Married customers dominate; Singles & Unmarried may be untapped.

![download](https://github.com/user-attachments/assets/2ba03577-580a-40c6-b917-19f66b52ae9b)

 Mostly Executives & Managers; AVP & VP are rare

 






 




