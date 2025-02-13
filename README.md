# Exploratory Data Analysis & Feature Engineering on Travel Dataset

![Project Logo](https://github.com/01-vivek-kumar/travel-eda-project1/blob/main/logo.png)

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
df.shape   
df.head() 
df.info()  
df.dtypes   
df.columns  

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

 


### 🔹 2. Univariate Analysis – Numerical Columns

To analyze the distribution of numerical variables, we used histograms. This helps in understanding the spread, skewness, and presence of outliers in the data.

A KDE (Kernel Density Estimate) curve was also added to visualize the probability distribution of each numerical feature smoothly.

🔹 Approach:
Histograms were plotted for each numerical column.
KDE plots were included to observe the underlying distribution.
The color scheme was kept consistent (skyblue) for better readability.

```python
# Univariate analysis for numerical columns
# Using histograms to understand the distribution

for num_column in nums:
    plt.figure(figsize=(8,5))
    sns.histplot(df[num_column], kde=True, color='skyblue')
    plt.title(f'Univariate Analysis - {num_column}')
    plt.show()

```


![download](https://github.com/user-attachments/assets/feac7fed-50d6-4c43-a064-fc7f21f3b1db)

**Insight:** The distribution of `CustomerID` appears fairly uniform, suggesting a balanced dataset without significant gaps or missing values.  

**Takeaway:** No immediate issues in customer ID allocation, and the dataset is well-represented for analysis.


![download](https://github.com/user-attachments/assets/c232196e-387d-49d1-a67c-c12eb9f6d209)

**Insight:** The age distribution is right-skewed, with most customers concentrated between 25-40 years old.  

**Takeaway:** The business primarily attracts a younger demographic, so marketing strategies can be tailored accordingly.


![download](https://github.com/user-attachments/assets/0bac51fb-e7a2-4b05-a3e1-48f01543ff1e)

![download](https://github.com/user-attachments/assets/d2d3ad29-9c9d-4b97-b95e-471f4f5f8fe8)


![download](https://github.com/user-attachments/assets/e3ee264f-e3a9-4e4c-a7b9-71be95e1b1ea)


![download](https://github.com/user-attachments/assets/e8b150c2-b7d1-4829-8e9a-fd32d55643e6)


![download](https://github.com/user-attachments/assets/ef45be29-80b8-4609-a212-abb8847d9049)


![download](https://github.com/user-attachments/assets/032f08c8-dfdb-4b6a-ba7f-d29cf7b01361)




### 🔹 3. Changing Data Types for Categorical Columns


During the data exploration phase, I observed that some numerical columns, such as CityTier, had mean values that didn’t make business sense. Since these columns represent categories rather than continuous numerical values, treating them as numerical would lead to incorrect insights.

Identified columns where mean values were misleading (CityTier, PreferredLoginDevice, Gender, PreferredPaymentMode).
Converted these columns from numerical to categorical data types using astype('category').

```python
df[cats] = df[cats].astype('object')
```

**After converting selected numerical columns to categorical, I rechecked the summary statistics using df[cats].describe().T. The output now provides relevant insights, such as:**

TypeofContact: Majority of the entries fall under "Self Enquiry" (2918 occurrences).
Occupation: Most customers are Salaried (1999 occurrences).
CityTier: Tier 1 has the highest representation (2678 occurrences).
Gender: Majority of customers are Male (2463 occurrences).

The categorical transformation ensured that the summary statistics now reflect meaningful categorical distributions rather than misleading numerical averages.

This step improves data interpretability and ensures that future visualizations and models treat categorical variables appropriately.



### 🔹 2. Bivariate analysis

After performing univariate analysis, where we examined individual variables, we now move to bivariate analysis to explore relationships between two variables. This helps us understand how one feature influences another, which is crucial for making business decisions.

**Age vs. DurationOfPitch**

```python
cross_tab = pd.crosstab(df['MaritalStatus'], df['ProdTaken'], normalize = 'index') #normalize give percentage

plt.bar(cross_tab.index, cross_tab[1], color = 'lightblue')

```
![download](https://github.com/user-attachments/assets/4f0e55d5-bd7a-460a-9a66-e4c0db8675e6)


Understanding this relationship can help tailor pitch duration based on age groups to improve conversion rates.


**MaritalStatus vs. ProdTaken**

```python
cross_tab.plot(kind = 'bar', stacked = True, color = ['lightblue', 'lightgreen'])
```

![download](https://github.com/user-attachments/assets/708e8082-dd12-4878-bd56-6da7ac331931)

If a specific marital status shows a higher product adoption rate, marketing efforts can be focused on that segment to optimize conversions.


** Insights - Single people taking more products **


**ProductPitched vs. PitchSatisfactionScore**

```python
sns.barplot(x = 'ProductPitched', y = 'PitchSatisfactionScore', data=df, palette = 'coolwarm')
```

![download](https://github.com/user-attachments/assets/fc302fa1-60ae-44b4-894f-b8eaa5a4b098)

Some product categories might have consistently higher satisfaction scores, indicating better customer reception.
If satisfaction scores are low for certain products, it may indicate a need for better sales strategies or product improvements.


**NumberOfFollowups vs. PitchSatisfactionScore**
```python
sns.lineplot(x = 'NumberOfFollowups', y = 'PitchSatisfactionScore', data = df, marker = 'o', color = 'purple')
```

![download](https://github.com/user-attachments/assets/1aed2ffe-1568-4ed6-a38d-f2cc3333ea0a)


Optimizing follow-ups is crucial—too many may irritate customers, while too few may miss opportunities.
The sales team should find the ideal number of follow-ups to maintain high satisfaction and maximize conversions.



### 🔹 3. Multivariate Analysis

Now moving to multivariate analysis, where we examine how multiple variables interact with each other. This helps uncover deeper relationships that may not be visible in univariate or bivariate analysis.

** Relationship Between Age, ProdTaken, and OwnCar**

```python
sns.violinplot(x='ProdTaken', y='Age', data=df, hue='OwnCar', palette='Set1')

```
![download](https://github.com/user-attachments/assets/7ed8703d-7b95-4ac9-ad47-26c18ae0e6cd)


**Relationship Between Gender, Age, and ProdTaken**

```python
sns.swarmplot(x='Gender', y='Age', data=df, hue='ProdTaken', palette='Set1')
```

![download](https://github.com/user-attachments/assets/922794cb-03e5-48b0-a93b-df0b94b220bf)


**Correlation Heatmap**

We use a heatmap to visualize relationships between numerical features.

```python
nums_corr = df[nums].corr()
sns.heatmap(nums_corr, annot=True, cmap='rainbow', fmt='.3f')

```
![download](https://github.com/user-attachments/assets/65b4cdb9-32a8-4313-8894-a7c1821860b4)


Strongly correlated features might be combined or used for feature selection.



## 🔄 Feature Engineering

Now, moving to Feature Engineering, where we modify or transform existing features to enhance model performance.

1: Dropping Irrelevant Features
CustomerID is a unique identifier and does not provide meaningful insights for modeling.

```python
df.drop('CustomerID', inplace=True, axis=1)

```

2: Creating New Features

To enhance the dataset, we created a new feature called TotalVisiting, which represents the total number of visitors (adults + children).
Combining NumberOfPersonVisiting and NumberOfChildrenVisiting provides a more holistic view of total visitors per customer.

```python
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']

```

 3: Dropping Redundant  Features
Since we created TotalVisiting, the original columns NumberOfPersonVisiting and NumberOfChildrenVisiting are no longer needed.

```python
df.drop(['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)
```

4: Separating Numerical and Categorical Features
Helps in applying appropriate transformations and scaling techniques later.

Numerical data consists of **continuous and discrete** values. While discrete values may sometimes be treated as categorical during analysis, for model training, all numerical features need to be **scaled** to ensure proper weightage and prevent larger values from dominating smaller ones. On the other hand, **categorical features** cannot be directly used in mathematical models since they contain non-numeric data. Hence, they must be **converted into numerical representations** using encoding techniques like **One-Hot Encoding (OHE) or Label Encoding**, allowing the model to interpret and learn from them effectively.

```python
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print("Number of Numerical Features are: ", len(num_features))

cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print("Number of categorical Features are: ", len(cat_features))
```

5: Train-Test Split

After separating numerical and categorical features, the next step is splitting the dataset into training and testing sets using train_test_split from sklearn.model_selection.


```python
from sklearn.model_selection import train_test_split

```

6. Defining features (X) and target variable (Y).

X: Contains all independent variables (features) used for prediction.
Y: Contains the target variable (ProdTaken), which we aim to predict.



```python
X = df.drop('ProdTaken', axis=1)  # Dropping target variable from features
Y = df['ProdTaken']  # Storing target variable separately
```

Splitting the Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```


7. Encoding and Scaling

Previously, we used a for loop to separate features. Now, we can use select_dtypes() for a more efficient approach:

```python
cat_features = X_train.select_dtypes(include=['object']).columns
num_features = X_train.select_dtypes(exclude=['object']).columns

print("Categorical Features:", list(cat_features))
print("Numerical Features:", list(num_features))
```

Now that we have separated categorical and numerical features, we need to preprocess them before feeding them into a machine learning model.

We can use ColumnTransformer from sklearn.compose to apply both transformations in one step:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop = 'first')
preprocessor = ColumnTransformer(
    [
        ('OnHotEncoder', oh_transformer, cat_features),
        ('StandardScaler', numeric_transformer, num_features)
    ]
)
```

Now, this preprocessor will be applied to X_train and X_test before model training! 


## 🔄 Key Results & Insights:

- Marital Status & Product Adoption: Married customers were more likely to take the product.
- Pitch Satisfaction & Product Adoption: Higher satisfaction led to a higher likelihood of purchase.
- Feature Importance: Variables like CityTier, TypeOfContact, and Occupation had a strong impact on customer decisions.
- Data Readiness: After preprocessing, my dataset was clean, structured, and optimized for training a predictive model.
- This project helped me solidify my understanding of data analysis, feature engineering, and preprocessing techniques, setting the foundation for model training.

## 🔄 Project Summary

Project Summary:
Data Understanding & Cleaning:

I identified missing values and handled them appropriately.
I changed incorrect data types and removed unnecessary columns like CustomerID.
Exploratory Data Analysis (EDA):

Univariate Analysis: Checked individual feature distributions.
Bivariate & Multivariate Analysis: Explored relationships between features using scatter plots, violin plots, and heatmaps.
Feature Engineering & Transformation:

Converted discrete numerical columns into categorical where necessary.
Created new features like TotalVisiting for better insights.
Encoded categorical variables and scaled numerical data for consistency.
Final Data Preparation:

Separated independent (X) and dependent (Y) variables.
Ensured data was ready for model building through encoding and scaling.


## 🔄 Contributing  
Since this project is a part of my learning journey, I'm open to feedback, suggestions, and collaboration to enhance it further. If you'd like to contribute:

Share insights & improvements – If you see areas where the analysis or modeling can be improved, feel free to suggest them.
Code contributions – If you're interested in refining the feature engineering, testing new models, or improving visualization, I'm happy to collaborate.
Discussions & learning together – I'm always open to learning new techniques and approaches, so feel free to reach out if you have suggestions or want to discuss the project.
This project is a work in progress, and I'm excited to keep refining it with new insights! 

## 🔄 Author - Vivek Kumar  

This project is part of my portfolio, showcasing the data analysis and machine learning skills essential for data science roles. If you have any questions, feedback, or would like to collaborate, feel free to get in touch!


## 🔄 Let's Connect
For more content on SQL, data analysis, and other data-related topics, make sure to follow me on social media:

LinkedIn: Connect with me professionally[(https://www.linkedin.com/in/vivek-kumar-vt/)]

Thank you for your support, and I look forward to connecting with you!

















