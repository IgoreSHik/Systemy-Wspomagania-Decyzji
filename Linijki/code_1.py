import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from itertools import islice

global points
global divisions

def right(data):
    global divisions
    sorted_data = sorted(data, key=lambda point: point[0])
    for i, entry in enumerate(islice(sorted_data, 1, None), start=1):
        if (sorted_data[i - 1][2] != entry[2]):
            coord = (sorted_data[i - 1][0]+entry[0])/2
            plt.axvline(x=coord, color='red', linestyle='--')
            sorted_data = sorted_data[i:]
            for entry in points:
                if (entry["x"]>coord):
                    entry["string"] = entry["string"] + " 1"
                else:
                    entry["string"] = entry["string"] + " 0"
            divisions+=1
            return up(sorted_data)
    return False

def up(data):
    global divisions
    sorted_data = sorted(data, key=lambda point: point[1])
    for i, entry in enumerate(islice(sorted_data, 1, None), start=1):
        if (sorted_data[i - 1][2] != entry[2]):
            coord = (sorted_data[i - 1][1]+entry[1])/2
            plt.axhline(y=coord, color='red', linestyle='--')
            sorted_data = sorted_data[i:]
            for entry in points:
                if (entry["y"]>coord):
                    entry["string"] = entry["string"] + " 1"
                else:
                    entry["string"] = entry["string"] + " 0"
            divisions+=1
            return left(sorted_data)
    return False

def left(data):
    global divisions
    sorted_data = sorted(data, key=lambda point: point[0], reverse=True)  
    for i, entry in enumerate(islice(sorted_data, 1, None), start=1):
        if (sorted_data[i - 1][2] != entry[2]):
            coord = (sorted_data[i - 1][0]+entry[0])/2
            plt.axvline(x=coord, color='blue', linestyle='--')
            sorted_data = sorted_data[i:]
            for entry in points:
                if (entry["x"]<coord):
                    entry["string"] = entry["string"] + " 1"
                else:
                    entry["string"] = entry["string"] + " 0"
            divisions+=1
            return down(sorted_data)
    return False

def down(data):
    global divisions
    sorted_data = sorted(data, key=lambda point: point[1], reverse=True)
    for i, entry in enumerate(islice(sorted_data, 1, None), start=1):
        if (sorted_data[i - 1][2] != entry[2]):
            coord = (sorted_data[i - 1][1]+entry[1])/2
            plt.axhline(y=coord, color='blue', linestyle='--')
            sorted_data = sorted_data[i:]
            for entry in points:
                if (entry["y"]<coord):
                    entry["string"] = entry["string"] + " 1"
                else:
                    entry["string"] = entry["string"] + " 0"
            divisions+=1
            return sorted_data
    return False

def browse_file():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Choose a file"
    )

    root.destroy()

    return file_path

file_path = browse_file()

if not file_path:
    exit()

data = []

with open(file_path, 'r') as file:
    for line in file:
        x, y, cls = map(str.strip, line.split(','))
        data.append((float(x), float(y), cls))

class_data_sorted = {}
divisions = 0

for x, y, cls in data:
    if cls not in class_data_sorted:
        class_data_sorted[cls] = {'x': [], 'y': []}
    class_data_sorted[cls]['x'].append(x)
    class_data_sorted[cls]['y'].append(y)

for i, (cls, values) in enumerate(class_data_sorted.items()):
    plt.scatter(values['x'], values['y'], label=cls)

points = []

for x, y, cls in data:
    point = {
        "x": float(x),
        "y": float(y),
        "class": cls,
        "string": ""
    }
    points.append(point)

result = right(data)
while result:
    print(len(result))
    result = right(result)

filename = 'output_data.txt'

with open(filename, 'w') as file:
    for point in points:
        file.write(str(point["x"])+", "+str(point["y"])+", "+point["class"]+","+point["string"] + '\n')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

print (divisions)

plt.show()