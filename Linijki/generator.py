import random
import csv

def generate_data(num_points):
    data = []
    for _ in range(num_points):
        x = round(random.uniform(-100, 100), 2)
        y = round(random.uniform(-100, 100), 2)
        class_label = random.choice(['A', 'B', 'C'])  # Замените классы на нужные вам
        data.append([x, y, class_label])
    return data

def save_to_csv(data, filename='generated_data.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Генерация данных
num_points = 10
generated_data = generate_data(num_points)

# Сохранение в CSV файл
save_to_csv(generated_data)

# Testy reczne
# test1 = []
# test1.append([0, 0, 'A'])
# test1.append([2, 0, 'A'])
# test1.append([0, 1, 'C'])
# test1.append([3, 1, 'A'])
# test1.append([1, 2, 'B'])
# test1.append([2, 2, 'C'])
# test1.append([0, 3, 'B'])
# save_to_csv(test1, 'test1.csv')

# test2 = []
# test2.append([0, 0, 'A'])
# test2.append([1, 0, 'B'])
# test2.append([2, 0, 'B'])
# test2.append([0, 1, 'B'])
# test2.append([1, 1, 'A'])
# test2.append([2, 1, 'A'])
# save_to_csv(test2, 'test2.csv')

# test3 = []
# test3.append([0, 0, 'C'])
# test3.append([3, 0, 'A'])
# test3.append([1, 1, 'B'])
# test3.append([2, 1, 'C'])
# save_to_csv(test3, 'test3.csv')

# test4 = []
# test4.append([0, 0, 'A'])
# test4.append([1, 0, 'A'])
# test4.append([2, 0, 'A'])
# test4.append([3, 0, 'A'])
# test4.append([4, 0, 'B'])
# save_to_csv(test4, 'test4.csv')

# test5 = []
# test5.append([0, 0, 'A'])
# test5.append([1, 0, 'B'])
# test5.append([2, 0, 'B'])
# test5.append([3, 0, 'A'])
# test5.append([0, 1, 'B'])
# test5.append([1, 1, 'A'])
# test5.append([2, 1, 'A'])
# test5.append([3, 1, 'B'])
# test5.append([0, 2, 'B'])
# test5.append([1, 2, 'A'])
# test5.append([2, 2, 'A'])
# test5.append([3, 2, 'B'])
# test5.append([0, 3, 'A'])
# test5.append([1, 3, 'B'])
# test5.append([2, 3, 'B'])
# test5.append([3, 3, 'A'])
# save_to_csv(test5, 'test5.csv')

# test6 = []
# test6.append([1, 0, 'C'])
# test6.append([4, 0, 'A'])
# test6.append([2, 1, 'B'])
# test6.append([3, 1, 'C'])
# test6.append([0, 2, 'A'])
# test6.append([5, 3, 'A'])
# save_to_csv(test6, 'test6.csv')

# test7 = []
# test7.append([0, 0, 'A'])
# test7.append([2, 1, 'B'])
# test7.append([1, 3, 'B'])
# test7.append([3, 2, 'A'])
# save_to_csv(test7, 'test7.csv')

# test8 = []
# test8.append([0, 0, 'A'])
# test8.append([0, 0, 'A'])
# test8.append([0, 0, 'A'])
# test8.append([0, 0, 'B'])
# test8.append([1, 0, 'B'])
# test8.append([1, 0, 'B'])
# test8.append([1, 0, 'B'])
# test8.append([1, 0, 'A'])
# save_to_csv(test8, 'test8.csv')

# test9 = []
# test9.append([0, 0, 'A'])
# test9.append([1, 0, 'A'])
# test9.append([2, 0, 'A'])
# test9.append([0, 1, 'A'])
# test9.append([1, 1, 'A'])
# test9.append([2, 1, 'B'])
# test9.append([0, 2, 'A'])
# test9.append([1, 2, 'B'])
# test9.append([2, 2, 'B'])
# test9.append([1, 3, 'B'])
# test9.append([2, 3, 'B'])
# test9.append([3, 3, 'B'])
# save_to_csv(test9, 'test9.csv')

# test10 = []
# test10.append([0, 0, 'A'])
# test10.append([1, 0, 'A'])
# test10.append([2, 0, 'A'])
# test10.append([1, 1, 'A'])
# test10.append([2, 1, 'A'])
# test10.append([3, 1, 'B'])
# test10.append([2, 2, 'A'])
# test10.append([3, 2, 'B'])
# test10.append([4, 2, 'B'])
# test10.append([3, 3, 'B'])
# test10.append([4, 3, 'B'])
# test10.append([5, 3, 'B'])
# save_to_csv(test10, 'test10.csv')

# test12 = []
# test12.append([0, 0, 0, 1, 'A'])
# test12.append([0, 0, 1, 0, 'B'])
# test12.append([0, 1, 0, 0, 'C'])
# test12.append([1, 0, 0, 0, 'D'])
# save_to_csv(test12, 'test12.csv')


