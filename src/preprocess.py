import pandas as pd
from src.load_data import load_raw_dataset
non_predictive_columns = ['state', 'county', 'community', 'communityname', 'fold']
missing_threshold = 0.5
def drop_non_predictive(df):
    return df.drop(columns = non_predictive_columns)
def drop_high_missing(df, threshold):
    missing_fraction = df.isna().mean()
    to_drop = missing_fraction[missing_fraction > threshold].index
    print(f"Dropping {len(to_drop)} columns")
    return df.drop(columns = to_drop)
def impute_remaining(df):
    df = df.fillna(df.median())
    return df
def clean_dataset(df):
    df = drop_non_predictive(df)
    df = drop_high_missing(df, missing_threshold)
    df = impute_remaining(df)
    return df
if __name__ == "__main__":
    df = load_raw_dataset("dataset/raw/communities.data", "dataset/raw/communities.names")
    df_clean = clean_dataset(df)
    df_clean.to_csv("dataset/processed/communities_clean.csv", index = False)
    print(df_clean.shape)
    print(df_clean.isna().sum().sum())
    print(df_clean.columns[-1])
    print(df_clean.dtypes.value_counts())