import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import ipaddress
import shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
print("--- Loading Datasets ---")
try:
    fraud_data_df = pd.read_csv('Fraud_Data.csv')
    ip_to_country_df = pd.read_csv('IpAddress_to_Country.csv')
    creditcard_df = pd.read_csv('creditcard.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: Ensure 'Fraud_Data.csv', 'IpAddress_to_Country.csv', and 'creditcard.csv' are in the same directory.")
    exit()

# --- 2. Data Analysis and Preprocessing - Fraud_Data.csv ---
print("\n--- Preprocessing Fraud_Data.csv ---")

# 2.1. Handle Missing Values
print("Checking for missing values in Fraud_Data.csv:")
print(fraud_data_df.isnull().sum())
# No explicit instruction to impute/drop, assuming no significant missing values based on typical datasets.
# If there were, common methods include:
# fraud_data_df.dropna(inplace=True) # Drop rows with any missing values
# fraud_data_df['age'].fillna(fraud_data_df['age'].median(), inplace=True) # Impute with median

# 2.2. Data Cleaning - Correct Data Types
print("Correcting data types for timestamps...")
fraud_data_df['signup_time'] = pd.to_datetime(fraud_data_df['signup_time'])
fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])
print("Timestamp columns converted.")

# 2.3. Data Cleaning - Remove Duplicates
print("Checking for duplicates in Fraud_Data.csv...")
initial_rows = fraud_data_df.shape
fraud_data_df.drop_duplicates(inplace=True)
rows_after_dedup = fraud_data_df.shape
print(f"Removed {initial_rows - rows_after_dedup} duplicate rows.") # [2, 3, 4, 5]

# 2.4. Exploratory Data Analysis (EDA) - Initial Observations
print("\n--- EDA for Fraud_Data.csv ---")
print("Class distribution:")
print(fraud_data_df['class'].value_counts(normalize=True)) # [6]
print("\nDescriptive statistics for purchase_value and age:")
print(fraud_data_df[['purchase_value', 'age']].describe()) # [6]

# 2.5. Feature Engineering - Time-Based Features for Fraud_Data.csv
print("Engineering time-based features...")
fraud_data_df['hour_of_day'] = fraud_data_df['purchase_time'].dt.hour # [7, 8, 9]
fraud_data_df['day_of_week'] = fraud_data_df['purchase_time'].dt.dayofweek # [10, 8, 9]
fraud_data_df['time_since_signup'] = (fraud_data_df['purchase_time'] - fraud_data_df['signup_time']).dt.total_seconds() / (60*60*24) # in days [8]

# 2.6. Feature Engineering - Transaction Frequency and Velocity for Fraud_Data.csv
print("Engineering transaction frequency and velocity features...")
# Sort by user_id and time for correct velocity calculation
fraud_data_df.sort_values(by=['user_id', 'purchase_time'], inplace=True)

# Transaction velocity (time since last transaction for the same user)
fraud_data_df['time_since_last_transaction'] = fraud_data_df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() # [7, 8]
# Fill NaN for first transaction of each user (no previous transaction)
fraud_data_df['time_since_last_transaction'].fillna(fraud_data_df['time_since_last_transaction'].median(), inplace=True)

# Transaction frequency (e.g., count of transactions in last 24 hours for a user)
# This is more complex and often requires rolling windows. For simplicity, we'll use a simpler aggregation.
# A more robust approach would involve a feature store or more advanced time-series feature engineering.
# For demonstration, let's calculate mean purchase value per device.
fraud_data_df['mean_purchase_value_device'] = fraud_data_df.groupby('device_id')['purchase_value'].transform('mean') # [7]
fraud_data_df['purchase_value_deviation_device'] = abs(fraud_data_df['purchase_value'] - fraud_data_df['mean_purchase_value_device']) # [7]

# 2.7. Merging Datasets for Geolocation Analysis
print("Performing geolocation analysis...")
# Convert IP addresses to integer format for efficient lookup [11, 12, 13, 14]
# Using a robust method for IP to integer conversion
def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except (ipaddress.AddressValueError, TypeError):
        return np.nan # Handle invalid IP addresses

# Apply conversion to ip_address column in fraud_data_df
# Note: fraud_data_df['ip_address'] is already float64, implying it might be pre-converted or has NaNs.
# Assuming it's already in a numerical format that can be directly compared or needs conversion from string.
# If it's string, apply: fraud_data_df['ip_address_int'] = fraud_data_df['ip_address'].apply(ip_to_int)
# For this dataset, 'ip_address' is float64, so we assume it's already numerical.
# We need to ensure it's compatible with the ranges in ip_to_country_df.
# Let's convert ip_to_country_df bounds to int for consistency.
ip_to_country_df['lower_bound_ip_address'] = ip_to_country_df['lower_bound_ip_address'].astype(int)
ip_to_country_df['upper_bound_ip_address'] = ip_to_country_df['upper_bound_ip_address'].astype(int)

# Merge using a custom function or merge_asof for range lookup [11, 12, 15, 16]
# This is a common but tricky merge. A simple loop is too slow.
# For efficiency, we can sort and use merge_asof or a custom binary search.
# Let's use a more direct approach for demonstration, assuming IP ranges are non-overlapping and sorted.
# This approach is simplified and might not be optimal for very large IP datasets.
# A more robust solution would involve a custom function to find the country for each IP.

# Create a temporary column for merging
fraud_data_df['ip_address_temp'] = fraud_data_df['ip_address'].astype(int)

# Sort both dataframes for merge_asof
fraud_data_df.sort_values('ip_address_temp', inplace=True)
ip_to_country_df.sort_values('lower_bound_ip_address', inplace=True)

# Perform merge_asof to find the country for each IP
# This will merge if ip_address_temp is >= lower_bound_ip_address
# and then we filter to ensure it's <= upper_bound_ip_address
fraud_data_df = pd.merge_asof(
    fraud_data_df,
    ip_to_country_df,
    left_on='ip_address_temp',
    right_on='lower_bound_ip_address',
    direction='forward' # Use 'forward' to find the next valid range
)

# Filter out IPs that are not within the found range (i.e., ip_address_temp > upper_bound_ip_address)
fraud_data_df = fraud_data_df[
    (fraud_data_df['ip_address_temp'] >= fraud_data_df['lower_bound_ip_address']) &
    (fraud_data_df['ip_address_temp'] <= fraud_data_df['upper_bound_ip_address'])
].copy() # Use.copy() to avoid SettingWithCopyWarning

# Drop temporary and redundant IP columns
fraud_data_df.drop(columns=['ip_address_temp', 'lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)
print("Geolocation information merged.")

# 2.8. Data Transformation - Encode Categorical Features for Fraud_Data.csv
print("Encoding categorical features...")
categorical_cols_fraud = ['source', 'browser', 'sex', 'country']
# Check if 'country' column exists after merge, if not, handle it.
if 'country' not in fraud_data_df.columns:
    print("Warning: 'country' column not found after IP merge. Skipping country encoding.")
    categorical_cols_fraud.remove('country')

# Use One-Hot Encoding [17, 18, 19, 20]
fraud_data_df = pd.get_dummies(fraud_data_df, columns=categorical_cols_fraud, drop_first=True)
print("Categorical features encoded.")

# 2.9. Data Transformation - Normalization and Scaling for Fraud_Data.csv
print("Scaling numerical features...")
numerical_cols_fraud = ['purchase_value', 'age', 'ip_address', 'time_since_signup',
                        'time_since_last_transaction', 'mean_purchase_value_device',
                        'purchase_value_deviation_device']

# Filter for columns that actually exist after previous steps
numerical_cols_fraud = [col for col in numerical_cols_fraud if col in fraud_data_df.columns]

scaler_fraud = StandardScaler() # [2, 21, 22, 23]
fraud_data_df[numerical_cols_fraud] = scaler_fraud.fit_transform(fraud_data_df[numerical_cols_fraud])
print("Numerical features scaled.")

# Drop original time columns and device_id, user_id as they are no longer needed as features
fraud_data_df.drop(columns=['signup_time', 'purchase_time', 'device_id', 'user_id'], inplace=True)
print("Original time and ID columns dropped.")

# --- 3. Data Analysis and Preprocessing - creditcard.csv ---
print("\n--- Preprocessing creditcard.csv ---")

# 3.1. Handle Missing Values
print("Checking for missing values in creditcard.csv:")
print(creditcard_df.isnull().sum()) # [24, 25]
# Dataset is known to have no missing values.

# 3.2. Data Cleaning - Remove Duplicates
print("Checking for duplicates in creditcard.csv...")
initial_rows_cc = creditcard_df.shape
creditcard_df.drop_duplicates(inplace=True)
rows_after_dedup_cc = creditcard_df.shape
print(f"Removed {initial_rows_cc - rows_after_dedup_cc} duplicate rows.") # [2, 3, 4, 5]

# 3.3. Exploratory Data Analysis (EDA) - Initial Observations
print("\n--- EDA for creditcard.csv ---")
print("Class distribution:")
print(creditcard_df['Class'].value_counts(normalize=True)) # [6]
print("\nDescriptive statistics for Amount and Time:")
print(creditcard_df].describe()) # [6]

# 3.4. Data Transformation - Normalization and Scaling for creditcard.csv
print("Scaling 'Time' and 'Amount' features...")
# 'Time' and 'Amount' are the only non-PCA features that need scaling.
# V1-V28 are already scaled by PCA.
scaler_cc = StandardScaler() # [2, 21, 22, 23]
creditcard_df] = scaler_cc.fit_transform(creditcard_df])
print("Time and Amount features scaled.")
