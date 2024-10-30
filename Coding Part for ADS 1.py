import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

# Load the dataset
data_path = 'heart.csv'
heart_df = pd.read_csv(data_path)

# 1. Function to inspect data with comprehensive statistics
def inspect_data(df):
    """
    Prints descriptive statistics, correlation matrix,
    skewness, and kurtosis for each numerical column.
    """
    print("Descriptive Statistics:")
    print(df.describe())  # Basic statistical summary
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    print("\nCorrelation Matrix:")
    print(numeric_df.corr())  # Correlation among numeric features
    
    # Skewness and Kurtosis to check data distribution
    print("\nSkewness of each numerical feature:")
    print(numeric_df.apply(lambda x: skew(x)))
    print("\nKurtosis of each numerical feature:")
    print(numeric_df.apply(lambda x: kurtosis(x)))

# 2. Preprocessing: Categorical Encoding and Scaling
def preprocess_data(df):
    """
    Encodes categorical features and scales numerical features for uniformity.
    """
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_encoded[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
        df_encoded[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])
    
    return df_encoded

# 3. Line Plot for Max Heart Rate by Age, color-coded by Heart Disease
def plot_maxhr_by_age(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Age', y='MaxHR', hue='HeartDisease', data=df, palette={0: 'blue', 1: 'red'}, marker="o")
    plt.title("Maximum Heart Rate by Age (Color-coded by Heart Disease)", fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Max Heart Rate", fontsize=12)
    plt.legend(title="Heart Disease", labels=['No', 'Yes'])
    plt.show()

# 4. Histogram for Age Distribution
def plot_age_distribution(df):
    plt.figure(figsize=(8, 6))
    plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
    plt.title("Age Distribution of Patients", fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.show()

# 5. Box Plot for Cholesterol Levels by Heart Disease Status
def plot_cholesterol_by_heart_disease(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='HeartDisease', y='Cholesterol', data=df, palette='coolwarm')
    plt.title("Cholesterol Levels by Heart Disease Status", fontsize=14, fontweight='bold')
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)", fontsize=12)
    plt.ylabel("Cholesterol (mg/dL)", fontsize=12)
    plt.show()

# Execution: Applying each function in sequence
inspect_data(heart_df)
heart_df_processed = preprocess_data(heart_df)
plot_maxhr_by_age(heart_df)
plot_age_distribution(heart_df)
plot_cholesterol_by_heart_disease(heart_df)