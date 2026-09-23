import pandas as pd
def get_column_names(names_path):
    columns = []
    with open(names_path, 'r') as f:
        for line in f:
            if line.startswith('@attribute'):
                column_name = line.split()[1]
                columns.append(column_name)
    return columns
def load_raw_dataset(data_path, names_path):
    column_names = get_column_names(names_path)
    df = pd.read_csv(data_path, header=None, names=column_names, na_values='?')
    return df
if __name__ == "__main__":
    columns = get_column_names("dataset/raw/communities.names")
    df = load_raw_dataset("dataset/raw/communities.data", "dataset/raw/communities.names")
    print(len(columns))
    print(df.shape)
    print(df.columns[-1])
    print(df["PolicPerPop"].isna().sum())
    print(df["OtherPerCap"].isna().sum())
    print(df.dtypes.value_counts())