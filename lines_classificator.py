import pandas as pd
import numpy as np
from itertools import combinations
import math
import matplotlib.pyplot as plt
import time
from collections import defaultdict


class LinesClassificator:
    
    def __init__(self):
        self.possible_splits_dict = None
        self.is_split_dict = None
        self.rewards = None
        self.splits = []

    
    def remove_duplicates(self, df):
        grouped = df.groupby(df.columns[:-1].tolist())
        cleaned_rows = []

        for _, group_data in grouped:
            most_frequent_label = group_data[df.columns[-1]].mode().iloc[0]
            group_data[df.columns[-1]] = most_frequent_label
            cleaned_rows.append(group_data)

        cleaned_df = pd.concat(cleaned_rows, ignore_index=True)
        return cleaned_df
    

    def drawLines(self, df):
        fig, ax = plt.subplots()
        unique_classes = df['class'].unique()
        colors = plt.cm.rainbow_r(np.linspace(0, 1, len(unique_classes)))
        for i, class_label in enumerate(unique_classes):
            class_data = df[df['class'] == class_label]
            ax.scatter(x=class_data[df.columns[0]], y=class_data[df.columns[1]], label=class_label, color=colors[i])
        ax.legend()
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        for split in self.splits:
            column = split[0]
            value = split[1]
            if column == 0:
                plt.axvline(x=value, color='blue', linestyle='--')
            if column == 1:
                plt.axhline(y=value, color='red', linestyle='--') 
             
        plt.show()
    

    def calculate_mistakes(self, df):
        classes = df['class']
        class_counts = defaultdict(int)

        for c in classes:
            class_counts[c] += 1
        
        n = len(df)

        same_class_pairs_sum = sum((count * (count - 1)) / 2 for count in class_counts.values())
        all_pairs_sum = (n * (n-1))/2
        mistakes = all_pairs_sum - same_class_pairs_sum
        
        return mistakes


    def find_best_cut(self, df, mistakes):
        best_dimension = None
        best_threshold = None
        best_mistakes = mistakes

        dimensions = len(df.columns) - 2
        
        for dimension in range(dimensions):
            data = df.sort_values(by = df.columns[dimension])

            data['midpoint'] = (data[data.columns[dimension]].shift(-1) + data[data.columns[dimension]]) / 2
            data['midpoint'] = data['midpoint'].fillna(data.iloc[-1][data.columns[dimension]])

            data['left_count_by_binary_and_class'] = data.groupby(['binary', 'class']).cumcount().values
            data['left_count_by_binary'] = data.groupby('binary').cumcount().values
            data['right_count_by_binary_and_class'] = data.groupby(['binary', 'class']).cumcount(ascending=False).values
            data['right_count_by_binary'] = data.groupby('binary').cumcount(ascending=False).values
            data['delta_mistakes'] = data['left_count_by_binary'] - data['left_count_by_binary_and_class'] - data['right_count_by_binary'] + data['right_count_by_binary_and_class']
            data['total_delta_mistakes'] = data['delta_mistakes'].cumsum()

            data = data.drop_duplicates(subset=[df.columns[dimension]], keep='last')
            best_row = data.nsmallest(1, 'total_delta_mistakes', keep='first')
            
            actual_mistakes = mistakes + best_row['total_delta_mistakes'].iat[0]

            if actual_mistakes < best_mistakes:
                best_mistakes = actual_mistakes
                best_dimension = dimension
                best_threshold = best_row['midpoint'].iat[0]

        return best_dimension, best_threshold, best_mistakes

    
    def fit(self, df):
        cleaned = self.remove_duplicates(df)
        cleaned.columns = [*cleaned.columns[:-1], 'class']
        cleaned['binary'] = ''

        mistakes = self.calculate_mistakes(cleaned)
        print(mistakes)

        start_loop_time = time.time()
        while mistakes > 0:
            
            dimension, threshold, mistakes = self.find_best_cut(cleaned, mistakes)
            print(dimension, threshold, mistakes)
            self.splits.append((dimension, threshold, mistakes))
            
            suffixes = cleaned[cleaned.columns[dimension]].apply(lambda x: '1' if x > threshold else '0')
            cleaned['binary'] = cleaned['binary'] + suffixes
        end_loop_time = time.time()

        print('Execution time: ', end_loop_time -start_loop_time, 'seconds')
        print("Number of cuts: ", len(self.splits))
        print("Dropped points: ", len(df) - len(cleaned))

        print(cleaned)
        cleaned.to_csv('output_lines.csv', index=False)

        if len(df.columns) - 1 == 2:
            self.drawLines(cleaned)

        return cleaned

    

    def predict(self, predict_df):
        predictions = []

        for split in self.splits:
            column = split[0]
            value = split[1]
            predictions.append((predict_df[predict_df.columns[column]] > value).astype(int).astype(str))

        return [''.join(binary) for binary in zip(*predictions)]



