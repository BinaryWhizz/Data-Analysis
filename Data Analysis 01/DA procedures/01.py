# Few steps : 

# 1. Load Data
# 2. Validate Data
# 3. Audit Data
# 4. Clean Data (ETL)
# 5. Feature Engineering
# 6. EDA
# 7. Visualization
# 8. Output



import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings


# 1 --- Building Mini Pipeline 

def load_data(path):
    df = pd.read_csv(path)
    return df


def audit_data(df):
    print(df.shape)
    print(df.isnull().sum())


def clean_data(df):
    df = df.drop_duplicates()
    return df


def main():
    df = load_data("data.csv")
    audit_data(df)
    df = clean_data(df)


if __name__ == "__main__":
    main()



# Functions = reusable blocks
# main() = controller
# Clean structure 




# 2 --- Upgrade ---

# Smart Missing Value Handling 

def handle_missing(df):
    for col in df.select_dtypes(include='number'):
        df[col] = df[col].fillna(df[col].median())
    
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna("Missing")
    
    return df


# Auto Column Detection

num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns


# Correlation Analysis 

def correlation_analysis(df):
    corr = df.corr()
    print(corr) 


# Visualization

def plot_histograms(df):
    df.hist(figsize=(10,6))
    plt.show()


# to be continued ............... 