import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

from knn import KNN, evaluate_classification, evaluate_all
from lines_classificator import LinesClassificator

global label_terminal
global opened_file
global df
global df_predict
global app
global file_path

def hist():
    column_name = column_entry.get()
    plt.title(f"Histogram dla zmiennej {column_name}")
    if (discretize_num_bins.get()==""):
        plt.hist(df[column_name])
    else:
        plt.hist(df[column_name], bins=int(discretize_num_bins.get()))
    plt.show()

def draw2D():
    global df

    column_name = column_entry.get()
    column_name2 = column_entry2.get()

    df = df.sort_values(column_name)
    fig, ax = plt.subplots()
    column_name_color = column_entry_color.get()

    if not column_name_color:
        plt.title("Wykres 2D bez kolorowania klasy")
        ax.scatter(x=df[column_name], y=df[column_name2])
    else:
        plt.title("Wykres 2D z kolorowaniem klasy")
        unique_classes = df[column_name_color].unique()
        colors = plt.cm.rainbow_r(np.linspace(0, 1, len(unique_classes)))

        for i, class_label in enumerate(unique_classes):
            class_data = df[df[column_name_color] == class_label]
            ax.scatter(x=class_data[column_name], y=class_data[column_name2], label=class_label, color=colors[i])

        ax.legend()

    plt.xlabel(column_name)
    plt.ylabel(column_name2)
    plt.show()

def draw3D():
    global df

    column_name = column_entry.get()
    column_name2 = column_entry2.get()
    column_name3 = column_entry3.get()

    df = df.sort_values(column_name)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    column_name_color = column_entry_color.get()

    if not column_name_color:
        plt.title("Wykres 3D bez kolorowania klasy")
        ax.scatter(xs=df[column_name], ys=df[column_name2], zs=df[column_name3])
    else:
        unique_classes = df[column_name_color].unique()
        colors = plt.cm.rainbow_r(np.linspace(0, 1, len(unique_classes)))

        for i, class_label in enumerate(unique_classes):
            plt.title("Wykres 3D z kolorowaniem klasy")
            class_data = df[df[column_name_color] == class_label]
            ax.scatter(xs=class_data[column_name], ys=class_data[column_name2], zs=class_data[column_name3], label=class_label, color=colors[i])

        ax.legend()

    ax.set_xlabel(column_name)
    ax.set_ylabel(column_name2)
    ax.set_zlabel(column_name3)

    plt.show()

def load_data():
    global df
    global file_path
    separator = separator_field.get()
    has_header = checkbox_var.get()
    if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=0 if has_header else None)
    else:
        if has_header:
            df = pd.read_csv(file_path, sep=separator)
        else:
            df = pd.read_csv(file_path, sep=separator, header=None)
            df = df.set_axis(list(map(str, range(len(df.columns)))), axis=1)
    display_csv()

def load_data_predict(file_path, separator, has_header=True):
    has_header = checkbox_var.get()
    global df_predict
    if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        df_predict = pd.read_excel(file_path, header=0 if has_header else None)
    else:
        if has_header:
            df_predict = pd.read_csv(file_path, sep=separator)
        else:
            df_predict = pd.read_csv(file_path, sep=separator, header=None)

def convert_text_column_to_numeric(is_sorted=True):
    global df
    column_name = column_entry.get()
    if is_sorted:
        mapping = {value: index for index, value in enumerate(sorted(df[column_name].unique()))}
    else:
        mapping = {value: index for index, value in enumerate(df[column_name].unique())}

    df[column_name] = df[column_name].map(mapping)
    display_csv()

# Pierwszy przedział jest obustronnie domknięty, kolejne przedziały są domknięte tylko prawostronnie
def discretize_numeric_column():
    global df
    column_name = column_entry.get()
    num_bins = int(discretize_num_bins.get())

    if df[column_name].dtype == object: # Jeśli typ został rozpoznany jako string, bo jest przecinek zamiast kropki to zamień na float
        df[column_name] = df[column_name].str.replace(',', '.').astype(float)

    df[column_name] = pd.cut(df[column_name], bins=num_bins, labels=False)
    display_csv()

def normalize_numeric_column():
    global df
    column_name = column_entry.get()
    df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    display_csv()

def change_interval_numeric_column():
    global df
    column_name = column_entry.get()
    new_min = int(min_entry.get())
    new_max = int(max_entry.get())
    column = df[column_name]
    df[column_name] = (column - column.min()) / (column.max() - column.min()) * (new_max - new_min) + new_min
    display_csv()

def percentile_range():
    global df
    column_name = column_entry.get()
    lower_percentile = int(min_entry.get())
    upper_percentile = int(max_entry.get())
    column = df[column_name]
    lower = column.quantile(lower_percentile / 100)
    upper = column.quantile(upper_percentile / 100)
    df = df[(column >= lower) & (column <= upper)]
    display_csv()

# Function to open a file dialog and select a CSV file
def open_csv_file():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        global opened_file
        opened_file = file_path
        thread = Thread(target=load_data)
        thread.start()

def open_predict_file():
    global df
    global df_predict
    global metric
    file_path = filedialog.askopenfilename()
    result = []
    if file_path:
        global opened_file
        opened_file = file_path
        load_data_predict(file_path, separator=separator_field.get())
        column_name = column_entry.get()
        knn_classifier = KNN(k=2, metric=metric)
        knn_classifier.fit(np.array(df.drop(columns=[column_name])), np.array(df[column_name]))
        new_instance = np.array(df_predict)
        predictions = knn_classifier.predict(new_instance)
        for i, prediction in enumerate(predictions):
            result.append("The predicted class for " + str(new_instance[i]) + " is: " + str(prediction) + "\n")
        show_info(result, metric)

# Function to display the contents of the selected CSV file
def display_csv():
    app.title(opened_file)
    text.delete(1.0, tk.END)  # Clear any previous text
    text.insert(tk.END, df.to_string())

def display_csv_limit(n):
    text.delete(1.0, tk.END)
    text.insert(tk.END, df.head(n).to_string())

def show_first():
    global opened_file
    try:
        display_csv_limit(int(entry.get()))
    except NameError:
        app.title("ERROR")

def show_info(info, metric_values=None):
    info_window = tk.Tk()
    info_window.title("INFO")
    info_window.geometry("500x500")

    info_text = tk.Text(info_window, wrap=tk.WORD, width=120, height=30)
    info_text.grid(row=0, column=0, rowspan = 20)

    info_text.delete(1.0, tk.END)
    info_text.insert(tk.END, info)

    # Otwieranie pliku tekstowego do zapisu (symbol 'w' oznacza operację zapisywania)
    if metric_values:
        with open(f'{metric_values}_results.txt', 'w') as file:
            # Zapisywanie tekstu w pliku
            for x in info:
                file.write(x)

def clasification_evaluation():
    global df
    metric = 'euclidean'
    result = []
    for k in range(1, len(df)):
        column_name = column_entry.get()
        X_train = np.array(df.drop(column_name, axis=1))
        Y_train = np.array(df[column_name])
        accuracy = evaluate_classification(X_train, y_train, k=k, metric=metric)
        result.append(f"Accuracy using leave-one-out with {k} neighboors: {accuracy}\n")

    show_info(result, metric)

def start_clasification_evaluation_optimized():
    label_terminal.config(text=f"Started counting...")
    thread = Thread(target=clasification_evaluation_optimized)
    thread.start()

def clasification_evaluation_optimized():
    global df
    global metric
    column_name = column_entry.get()
    X_train = np.array(df.drop(column_name, axis=1))
    Y_train = np.array(df[column_name])
    accuracies = evaluate_all(app, label_terminal, X_train, Y_train, metric=metric)
    results = [f"Accuracy using leave-one-out with {k} neighboors: {accuracies[k]}\n" for k in range(1, len(df))]
    show_info(results, metric)
    x_values = [k for k in range(1, len(df))]
    y_values = [accuracies[k] for k in range(1, len(df))]
    plt.plot(x_values, y_values, label=metric)
    plt.xlabel('K') 
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('metric_visualisation.png')
    plt.show()


def fit_lines_classificator():
    global df
    global lines_classifier
    lines_classifier = LinesClassificator()
    df = lines_classifier.fit(df)
    display_csv()

def predict_lines_classificator():
    global df
    global df_predict

    file_path = filedialog.askopenfilename()
    if file_path:
        global opened_file
        opened_file = file_path
        load_data_predict(file_path, separator=separator_field.get())
        predictions = lines_classifier.predict(df_predict)
        show_info(predictions)

# Create the main application window
app = tk.Tk()
app.title("CSV Reader")
app.geometry("900x700")

label = tk.Label(app, text="Separator:")
label.grid(row=0, column=0)
separator_field = tk.Entry(app)
separator_field.grid(row=1, column=0)

# Create a button to open a CSV file
open_button = tk.Button(app, text="Open CSV File", command=open_csv_file)
open_button.grid(row=1, column=1)

global checkbox_var
checkbox_var = tk.BooleanVar()
checkbox = tk.Checkbutton(app, text="Dane maja naglowki", variable=checkbox_var)
checkbox.grid(row=2, column=1)

label = tk.Label(app, text="Column:")
label.grid(row=3, column=0)
column1 = tk.StringVar()
column_entry = tk.Entry(app, textvariable=column1)
column_entry.grid(row=4, column=0)
column_entry2 = tk.Entry(app)
column_entry2.grid(row=5, column=0)
column_entry3 = tk.Entry(app)
column_entry3.grid(row=6, column=0)
numeric_button = tk.Button(app, text="Change to numeric", command=convert_text_column_to_numeric)
numeric_button.grid(row=4, column=1)

label = tk.Label(app, text="Num_bins:")
label.grid(row=7, column=0)
discretize_num_bins = tk.Entry(app)
discretize_num_bins.grid(row=8, column=0)
discretize_button = tk.Button(app, text="Discretize", command=discretize_numeric_column)
discretize_button.grid(row=7, column=1)

normalize_button = tk.Button(app, text="Normalize", command=normalize_numeric_column)
normalize_button.grid(row=8, column=1)

label = tk.Label(app, text="Min/Lower:")
label.grid(row=9, column=0)
min_entry = tk.Entry(app)
min_entry.grid(row=10, column=0)
label = tk.Label(app, text="Max/Upper:")
label.grid(row=11, column=0)
max_entry = tk.Entry(app)
max_entry.grid(row=12, column=0)
interval_button = tk.Button(app, text="Change interval", command=change_interval_numeric_column)
interval_button.grid(row=13, column=1)

interval_button = tk.Button(app, text="Percentile range", command=percentile_range)
interval_button.grid(row=12, column=1)

label = tk.Label(app, text="First:")
label.grid(row=13, column=0)
entry = tk.Entry(app)
entry.grid(row=14, column=0)
filter_button = tk.Button(app, text="Show", command=show_first)
filter_button.grid(row=14, column=1)

label = tk.Label(app, text="Color:")
label.grid(row=15, column=0)
column_entry_color = tk.Entry(app)
column_entry_color.grid(row=16, column=0)

draw_button = tk.Button(app, text="Draw2D", command=draw2D)
draw_button.grid(row=16, column=1)

draw_button = tk.Button(app, text="Histogram", command=hist)
draw_button.grid(row=17, column=1)

draw_button = tk.Button(app, text="Draw3D", command=draw3D)
draw_button.grid(row=18, column=1)

knn_button = tk.Button(app, text="KNN Predict", command=open_predict_file)
knn_button.grid(row=19, column=1)

evaluate_button = tk.Button(app, text="Evaluate", command=start_clasification_evaluation_optimized)
evaluate_button.grid(row=20, column=1)

def update_metric(value):
    global metric
    metric = value

global metric
metric = "euclidean"
var = tk.StringVar(value="euclidean")
R1 = tk.Radiobutton(app, text="Metryka euklidesowa", variable=var, value="euclidean", command=lambda: update_metric("euclidean"))
R1.grid(row=21, column=1)

R2 = tk.Radiobutton(app, text="Metryka Manhattan", variable=var, value="manhattan", command=lambda: update_metric("manhattan"))
R2.grid(row=22, column=1)

R3 = tk.Radiobutton(app, text="Metryka Czebyszewa", variable=var, value="chebyshev", command=lambda: update_metric("chebyshev"))
R3.grid(row=23, column=1)

R4 = tk.Radiobutton(app, text="Metryka Mahalanobisa", variable=var, value="mahalanobis", command=lambda: update_metric("mahalanobis"))
R4.grid(row=24, column=1)

# Lines classificator buttons
fit_lines_button = tk.Button(app, text="Fit lines classificator", command=fit_lines_classificator)
fit_lines_button.grid(row=25, column=1)
predict_lines_button = tk.Button(app, text="Predict with lines classificator", command=predict_lines_classificator)
predict_lines_button.grid(row=26, column=1)
# Create a text widget to display CSV data
text = tk.Text(app, wrap="none", width=120, height=30)
text.grid(row=0, column=2, rowspan = 25, sticky="nsew")

# Создаем горизонтальный скроллбар
horizontal_scrollbar = tk.Scrollbar(app, orient="horizontal", command=text.xview)
horizontal_scrollbar.grid(row=25, column=2, sticky="ew")

# Привязываем горизонтальный скроллбар к текстовому полю
text.config(xscrollcommand=horizontal_scrollbar.set)

label_terminal = tk.Label(app, text="")
label_terminal.grid(row=26, column=2)

# Start the main event loop
app.mainloop()