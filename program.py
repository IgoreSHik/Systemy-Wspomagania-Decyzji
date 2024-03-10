import pandas as pd
import matplotlib.pyplot as plt

# Dane muszą byc oddzielone odpowiednim separatorem, nie może być komentarzy w pliku
def load_data(file_path, separator, has_header=True):
    if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=0 if has_header else None)
    else:
        if has_header:
            df = pd.read_csv(file_path, sep=separator)
        else:
            df = pd.read_csv(file_path, sep=separator, header=None)
    return df

def convert_text_column_to_numeric(df, column_name, is_sorted=True):
    if is_sorted:
        mapping = {value: index for index, value in enumerate(sorted(df[column_name].unique()))}
    else:
        mapping = {value: index for index, value in enumerate(df[column_name].unique())}

    df[column_name] = df[column_name].map(mapping)
    return df

# Pierwszy przedział jest obustronnie domknięty, kolejne przedziały są domknięte tylko prawostronnie
def discretize_numeric_column(df, column_name, num_bins):
    if df[column_name].dtype != float: # Jeśli typ został rozpoznany jako string, bo jest przecinek zamiast kropki to zamień na float
        df[column_name] = df[column_name].str.replace(',', '.').astype(float)

    df[column_name] = pd.cut(df[column_name], bins=num_bins, labels=False)
    return df

def normalize_numeric_column(df, column_name):
    df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    return df

def change_interval_numeric_column(df, column_name, new_min, new_max):
    column = df[column_name]
    df[column_name] = (column - column.min()) / (column.max() - column.min()) * (new_max - new_min) + new_min
    return df

def percentile_range(df, column_name, lower_percentile, upper_percentile):
    column = df[column_name]
    lower = column.quantile(lower_percentile / 100)
    upper = column.quantile(upper_percentile / 100)
    df = df[(column >= lower) & (column <= upper)]
    return df

def test():
    file_path = "przykladowe_dane.xlsx"
    data = load_data(file_path, separator="\t", has_header=True)
    data = convert_text_column_to_numeric(data, 'Hrabstwo', False)
    data = normalize_numeric_column(data, 'Przych')
    data = change_interval_numeric_column(data, 'Przych', 5, 10)
    data = percentile_range(data, 'Przych', 90, 95)
    print(data)

test()