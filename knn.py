from collections import Counter
import math
import numpy as np
import time

class KNN:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    def manhattan_distance(self, point1, point2):
        return sum(abs(x - y) for x, y in zip(point1, point2))

    def chebyshev_distance(self, point1, point2):
        return max(abs(x - y) for x, y in zip(point1, point2))

    def mahalanobis_distance(self, point1, point2):
        cov_inv = np.linalg.pinv(np.cov(np.vstack([self.X_train, point1, point2]).T))
        diff = np.array(point1) - np.array(point2)
        return math.sqrt(np.dot(np.dot(diff, cov_inv), diff))

    def calculate_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return self.euclidean_distance(point1, point2)
        elif self.metric == 'manhattan':
            return self.manhattan_distance(point1, point2)
        elif self.metric == 'chebyshev':
            return self.chebyshev_distance(point1, point2)
        elif self.metric == 'mahalanobis':
            return self.mahalanobis_distance(point1, point2)
        else:
            raise ValueError("Unsupported metric")

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = [self.calculate_distance(test_point, train_point) for train_point in self.X_train]
            sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
            k_nearest_labels = [self.y_train[i] for i in sorted_indices[:self.k]]
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions

    def predict_all(self, X_test):
        all_distances = [[self.calculate_distance(test_point, train_point) for train_point in self.X_train] for test_point in X_test]
        sorted_indices = [sorted(range(len(distances)), key=lambda k: distances[k]) for distances in all_distances]
        k_nearest_labels = [[self.y_train[i] for i in indices[:self.k]] for indices in sorted_indices]

        predictions = []        
        labels = k_nearest_labels[0]
        best_label = ""
        best_count = 0
        counter = {}

        for label in labels:
            counter[label] = counter.get(label, 0) + 1
            if counter[label] > best_count:
                best_count = counter[label]
                best_label = label
            predictions.append(best_label)

        return predictions

def evaluate_classification(X, Y, k, metric='euclidean'):
    correct_predictions = 0

    for i in range(len(X)):
        # X[i] and y[i] is test entry (leave-one-out)
        X_test = X[i]
        Y_test = Y[i]
        X_train = np.delete(X, i, 0)
        Y_train = np.delete(Y, i, 0)

        knn_classifier = KNN(k=k, metric=metric)
        knn_classifier.fit(X_train, Y_train)

        predicted_class = knn_classifier.predict([X_test])[0]
        if predicted_class == Y_test:
            correct_predictions += 1

    accuracy = correct_predictions / len(X)
    return accuracy

def evaluate_all(app, label_terminal, X, Y, metric='euclidean'):
    correct_predictions = [0] * (len(X)+1)
    accuracies = [0] * (len(X)+1)

    for i in range(len(X)):
        start = time.time()
        X_test = X[i]
        Y_test = Y[i]
        X_train = np.delete(X, i, 0)
        Y_train = np.delete(Y, i, 0)

        knn_classifier = KNN(k=len(X)-1, metric=metric)
        knn_classifier.fit(X_train, Y_train)

        predicted_classes = knn_classifier.predict_all([X_test])
        for k in range(1, len(X)):
            if predicted_classes[k-1] == Y_test:
                correct_predictions[k] += 1
        end = time.time()
        print(f"k = {i} time: {end-start}")
        label_terminal.config(text=f"k = {i} time: {end-start}")
    
    for k in range(1, len(X)):
        accuracies[k] = correct_predictions[k] / len(X)
    return accuracies


if __name__ == "__main__":
    # Przykladowe dane, trzeba podpiac dane od GUI
    X_train = [[1, 2], [2, 3], [3, 1], [4, 4]]
    y_train = ['A', 'A', 'B', 'B']

    # Pzyklad klasyfikacji pojedynczego obiektu
    knn_classifier = KNN(k=2, metric='euclidean')
    knn_classifier.fit(X_train, y_train)
    new_instance = [2.5, 2.5]
    predicted_class = knn_classifier.predict([new_instance])[0]
    print(f"The predicted class for the new instance is: {predicted_class}")

    # Przyklad oceny jakosci klasyfikacji
    for k in [1, 2, 3]:
        accuracy = evaluate_classification(X_train, y_train, k=k, metric='euclidean')
        print(f"Accuracy using leave-one-out with {k} neighboors: {accuracy}\n")