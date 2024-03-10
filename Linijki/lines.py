import pandas as pd
import numpy as np
from itertools import combinations
import math
import matplotlib.pyplot as plt

file_path = 'test12.csv'
if not file_path:
    exit()


data = pd.read_csv(file_path, header=None)

def remove_duplicates(df):
    grouped = df.groupby(df.columns[:-1].tolist())
    cleaned_df = pd.DataFrame(columns=df.columns)

    for _, group_data in grouped:
        most_frequent_label = group_data[df.columns[-1]].mode().iloc[0]
        selected_row = group_data.iloc[0]
        selected_row[df.columns[-1]] = most_frequent_label
        cleaned_df = pd.concat([cleaned_df, selected_row.to_frame().T])
    return cleaned_df.reset_index(drop=True)

cleaned = remove_duplicates(data)
cleaned.columns = [*cleaned.columns[:-1], 'class']
cleaned['binary'] = ''

print(cleaned)
print(f'Usuniete punkty: {len(data)-len(cleaned)}')

def calculate_possible_splits(df):
    columns_to_check = df.columns[:-2]
    possible_splits_dict = {}
    is_split_dict = {}

    for column in columns_to_check:
        unique_values = sorted(df[column].unique())
        possible_splits = [(unique_values[i] + unique_values[i+1])/2 for i in range(0, len(unique_values)-1)]
        possible_splits_dict[column] = possible_splits
        is_split_dict[column] = [False] * len(possible_splits)

    return possible_splits_dict, is_split_dict

possible_splits_dict, is_split_dict = calculate_possible_splits(cleaned)
print('------')
print(possible_splits_dict)
print(is_split_dict)

def calculate_rewards(df):
    n = len(df.columns) - 2
    eps = 1e-6
    array_data = df.iloc[:, :-2].to_numpy()
    rewards = {}

    for (row1_idx, row1), (row2_idx, row2) in combinations(enumerate(array_data), 2):
        count_same_columns = np.sum(row1 == row2)
        class_factor = 1.0 if df.loc[row1_idx, 'class'] != df.loc[row2_idx, 'class'] else -0.25
        key = (row1_idx, row2_idx)
        rewards[key] = 1 / (np.log(n - count_same_columns) + eps) * class_factor
        key2 = (row2_idx, row1_idx)
        rewards[key2] = rewards[key]
    return rewards

rewards = calculate_rewards(cleaned)
print('------')
print(rewards)

def stop_condition(df):
    last_column = df.columns[-1]
    pre_last_column = df.columns[-2]
    grouped = df.groupby(df[last_column])

    for group, group_data in grouped:
        unique_values_count = group_data[pre_last_column].nunique()
        if unique_values_count > 1:
            print(f"Not unique values in for group {group}")
            return False
    return True


def calculate_fitness(df, column, min_value, max_value, mid_value):
    left_indexes = df[(df[column] > min_value) & (df[column] < mid_value)].index.tolist()
    right_indexes = df[(df[column] > mid_value) & (df[column] < max_value)].index.tolist()

    fitness = 0.0
    for left_index in left_indexes:
        left_binary = df.loc[left_index, 'binary']
        for right_index in right_indexes:
            right_binary = df.loc[right_index, 'binary']
            if left_binary == right_binary:
                fitness += rewards[(left_index, right_index)]
    return fitness
            
def make_cut(df):
    columns_to_check = df.columns[:-2]
    
    best_column = None
    best_index = None
    best_fitness = -(math.inf)

    for column in columns_to_check:
        possible_splits = possible_splits_dict[column]
        is_split = is_split_dict[column]
        m = len(possible_splits)
        min_index = -1
        max_index = m
        # Init max_index
        for i in range(m):
            if is_split[i] == True:
                max_index = i
                break
        # Gasienica
        for i in range(m):
            if is_split[i]:
                min_index = i
                max_index = m
                for j in range(i+1, m):
                    if is_split[j] == True:
                        max_index = j
                        break
            else:
                min_value = possible_splits[min_index] if min_index >= 0 else -math.inf
                max_value = possible_splits[max_index] if max_index < m else math.inf
                mid_value = possible_splits[i]
                mid_index = i
                fitness = calculate_fitness(df, column, min_value, max_value, mid_value)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_column = column
                    best_index = mid_index
    
    best_value = possible_splits_dict[best_column][best_index]
    #Logi
    is_split_dict[best_column][best_index] = True
    splits.append((best_column, best_value, best_fitness))
    # Aktualizacja binary
    suffixes = df[best_column].apply(lambda x: '1' if x > best_value else '0')
    df['binary'] = df['binary'] + suffixes
    return df

splits = []
cuts = 0
while not stop_condition(cleaned):
    cleaned = make_cut(cleaned)
    cuts += 1
        

print(f'Liczba wykonanych ciec: {cuts}')
print(splits)
print(cleaned)

cleaned.to_csv('output_lines.csv', index=False)

def drawLines(df):

    fig, ax = plt.subplots()
    unique_classes = df['class'].unique()
    colors = plt.cm.rainbow_r(np.linspace(0, 1, len(unique_classes)))
    for i, class_label in enumerate(unique_classes):
        class_data = df[df['class'] == class_label]
        ax.scatter(x=class_data[df.columns[0]], y=class_data[df.columns[1]], label=class_label, color=colors[i])
    ax.legend()
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    for split in splits:
        column = split[0]
        value = split[1]
        if column == df.columns[0]:
           plt.axvline(x=value, color='blue', linestyle='--')
        if column == df.columns[1]:
           plt.axhline(y=value, color='red', linestyle='--')  
    plt.show()

n = len(cleaned.columns) - 2
if n == 2:
    drawLines(cleaned)


