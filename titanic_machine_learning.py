# -*- coding: utf-8 -*-
"""
Titanic - Machine Learning from Disaster (Refurnished)

This script analyzes the Titanic dataset to predict passenger survival.
It includes:
1. Data Loading (Local or Kaggle)
2. Data Preprocessing (Concatenation & Cleaning)
3. Exploratory Data Analysis (EDA) - Univariate, Bivariate, Multivariate
4. Feature Engineering
5. Encoding & Scaling
6. PCA & Modeling (Logistic Regression)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn modules
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualization style
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries Imported Successfully!")

# ==========================================
# 1. Loading the Datasets
# ==========================================
# USER: Update this path to where your data files are located
DATA_PATH = ""  # e.g., "C:/Users/Name/Downloads/titanic/"

import os

try:
    train_path = os.path.join(DATA_PATH, "/content/train.csv")
    test_path = os.path.join(DATA_PATH, "/content/test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print(f"Error: csv files not found in '{DATA_PATH}'. Please check the path.")
    exit()

# ==========================================
# 2. Concatenation of Datasets
# ==========================================
print("\n--- Concatenating Train and Test ---")
train_len = len(train_df)
passenger_ids = test_df['PassengerId']

# Combine
df = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0).reset_index(drop=True)
print(f"Combined DataFrame Shape: {df.shape}")


# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================
print("\n=== Exploratory Data Analysis ===")

# Create images directory
if not os.path.exists('images'):
    os.makedirs('images')
print("Created 'images' directory for plots.")

# --- 3.1 Univariate Analysis ---
print("\n--- 3.1 Univariate Analysis ---")
print(df.info())
print(df.describe())

# Pie Chart for Target (Train only)
plt.figure(figsize=(6, 6))
train_df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], explode=[0.05, 0], shadow=True)
plt.title('Survival Percentage (0=No, 1=Yes)')
plt.ylabel('')
plt.savefig('images/01_survival_pie.png')
plt.close()

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Age'].dropna(), kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Age Distribution')
sns.histplot(df['Fare'].dropna(), kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Fare Distribution')
plt.savefig('images/02_histograms.png')
plt.close()

# Countplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x='Sex', data=df, ax=axes[0])
sns.countplot(x='Embarked', data=df, ax=axes[1])
sns.countplot(x='Pclass', data=df, ax=axes[2])
plt.suptitle('Categorical Counts')
plt.savefig('images/03_countplots.png')
plt.close()

# Boxplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x=df['Age'], ax=axes[0], color='skyblue')
axes[0].set_title('Age Boxplot')
sns.boxplot(x=df['Fare'], ax=axes[1], color='salmon')
axes[1].set_title('Fare Boxplot')
plt.savefig('images/04_boxplots.png')
plt.close()


# --- 3.2 Bivariate Analysis ---
print("\n--- 3.2 Bivariate Analysis ---")

# Survival vs Sex
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title('Survival by Gender')
plt.savefig('images/05_survival_sex.png')
plt.close()

# Survival vs Pclass
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Survival by Passenger Class')
plt.savefig('images/06_survival_pclass.png')
plt.close()

# Age vs Survived (Violin)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Survived', y='Age', data=train_df, split=True)
plt.title('Age Distribution by Survival')
plt.savefig('images/07_survival_age.png')
plt.close()

# Fare vs Survived (Box)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=train_df)
plt.ylim(0, 300)
plt.title('Fare Distribution by Survival')
plt.savefig('images/08_survival_fare.png')
plt.close()


# --- 3.3 Multivariate Analysis ---
print("\n--- 3.3 Multivariate Analysis ---")

# Pairplot
cols_to_plot = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived']
sns.pairplot(train_df[cols_to_plot].dropna(), hue='Survived', palette='husl')
plt.savefig('images/09_pairplot.png')
plt.close()

# Heatmap
numeric_df = train_df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('images/10_heatmap.png')
plt.close()


# ==========================================
# 4. Data Preprocessing
# ==========================================

# 4.1 Handling Missing Values
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4.2 Feature Engineering: Title
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# 4.3 Encoding
label_enc = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title']:
    df[col] = label_enc.fit_transform(df[col].astype(str))

# ==========================================
# 5. Splitting, Scaling, PCA
# ==========================================
X = df.iloc[:train_len]
X_test_final = df.iloc[train_len:]
y = train_df['Survived']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"\nPCA Components: {X_pca.shape[1]}")

# ==========================================
# 6. Model Training (Logistic Regression)
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

print("\nTraining Logistic Regression...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))

# Submission
final_predictions = model.predict(X_test_pca)
submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": final_predictions})
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved to submission.csv")