import csv
import math
import numpy as np
import matplotlib.pyplot as plt

class DataModel():

    data = None
    sorted_class_data = None
    unique_classes = None

    def __init__(self, data = None):
        self.data = data
        if self.data is not None:
            self.unique_classes = self.find_unique_classes(self.data)
            self.sorted_class_data = self.sort_data_by_class(self.data)

    def load(self, filename=None):
        temp_data = self.get_list_from_csv(filename)
        self.data = self.preprocess_data(temp_data)
        self.unique_classes = self.find_unique_classes(self.data)
        self.sorted_class_data = self.sort_data_by_class(self.data)

    def get_list_from_csv(self, filename):
        file = open(filename, "r", encoding='utf-8-sig')
        lines = csv.reader(file)
        dataset = list(lines)
        file.close()
        return dataset

    def preprocess_data(self, original_data):
        for row in original_data:
            row = [float(cell) for cell in row]
        processed_data = np.asarray(original_data, dtype=np.float32)
        return processed_data

    def find_unique_classes(self, dataset):
        unique_classes = np.unique(dataset[:, -1])
        return unique_classes

    def sort_data_by_class(self, dataset):
        sortedclassdata = []
        for item in self.unique_classes:
            itemindex = np.where(dataset[:, -1] == item)
            singleclassdataset = dataset[itemindex, 0:np.size(dataset, 1) - 1]
            sortedclassdata.append(np.matrix(singleclassdataset))
        return sortedclassdata

class KDE():
    """Non parametric Kernel Density Estimator / Classifier. Allows user to
    input bandwidth (h, standard deviation of Gaussian components),but does not
    find it. Can classify for N dimensions, but only plot class / decision
    boundaries for 2."""

    training_model = DataModel()
    testing_model = DataModel()
    active_model = None
    bandwidth = 0.5

    def get_training_accuracy(self):
        """Returns the training accuracy of the classifier."""

        training_data = self.training_model.data
        total_count_of_data_points = len(training_data)

        total_correct_predictions = 0
        for i in range(total_count_of_data_points):

            correct_classification = training_data[i][-1]
            point = np.delete(training_data, -1, axis=1)[i]
            training_data_without_point = np.delete(training_data, i, axis=0)

            self.active_model = DataModel(training_data_without_point)

            if self.get_classification(point) == correct_classification:
                total_correct_predictions = total_correct_predictions + 1

        accuracy = 100 * (total_correct_predictions / total_count_of_data_points)
        return accuracy

    def get_testing_accuracy(self):
        """Returns the testing accuracy of the classifier."""

        self.active_model = DataModel(self.training_model.data)
        testing_data = self.testing_model.data
        total_count_of_data_points = len(testing_data)

        total_correct_predictions = 0
        for i in range(total_count_of_data_points):
            correct_classification = testing_data[i][-1]
            point = np.delete(testing_data, -1, axis=1)[i]
            if self.get_classification(point) == correct_classification:
                total_correct_predictions = total_correct_predictions + 1
        accuracy = 100 * (total_correct_predictions / total_count_of_data_points)
        return accuracy

    def get_classification(self, point):
        """Use kernal density model to classify point."""

        dataset = self.active_model.sorted_class_data
        unique_classes = self.active_model.unique_classes
        bandwidth = self.bandwidth

        probability_of_each_class = np.zeros(shape=(len(dataset), 1))

        num_classes = len(dataset)
        dimensions = dataset[0][0].shape[1]

        for class_index in range(num_classes):
            class_probability = 0
            sum_of_probabilities = 0
            total_count_of_points_in_this_class = len(dataset[class_index])
            for point_index in range(total_count_of_points_in_this_class):
                exponent_numerator = np.linalg.norm(point-dataset[class_index][point_index])
                exponent_denominator = bandwidth
                exponent = -0.5 * ( (exponent_numerator / exponent_denominator) ** 2)
                base = 1 / ((2 * math.pi * (bandwidth ** 2)) ** (dimensions / 2))
                new_probability = base * math.exp(exponent)
                sum_of_probabilities = sum_of_probabilities + new_probability
            class_probability = sum_of_probabilities / total_count_of_points_in_this_class
            probability_of_each_class[class_index] = class_probability

        class_prediction = unique_classes[np.argmax(probability_of_each_class)]
        return class_prediction

    def plot_class_boundaries(self, model = None):
        """Plot the class boundaries for 2d data."""

        if model is None:
            model = self.training_model

        if (len(model.data[0]) - 1) != 2:
            print("Can only plot models with 2 features/dimensions.")
            return

        self.active_model = model

        data_points = self.create_data_points()

        classes_for_data_points = []
        for i in range(len(data_points)):
            point = np.delete(data_points, -1, axis=1)[i]
            classes_for_data_points = np.append(classes_for_data_points,
                                                self.get_classification(point))
        data_points[:, 2] = classes_for_data_points
        self.plot(data_points)

    def create_data_points(self, points_per_axis=25):
        """Creates lots of points we can later classify and plot to see boundaries."""

        x_min = math.floor(min(self.active_model.data[:,0]))
        x_max = math.ceil(max(self.active_model.data[:,0]))
        y_min = math.floor(min(self.active_model.data[:,1]))
        y_max = math.ceil(max(self.active_model.data[:,1]))

        x_increment = (x_max - x_min) / points_per_axis
        y_increment = (y_max - y_min) / points_per_axis

        data = []
        for i in range(points_per_axis):
            for j in range(points_per_axis):
                x_value = x_min + (i * x_increment)
                y_value = y_min + (j * y_increment)
                data.append([x_value, y_value, 0])

        data = np.asarray(data)
        return data

    def plot(self, classified_data_points):
        """Plots the data color coded for two classes."""

        class_one_x = [row[0] for row in classified_data_points if row[2] == 0]
        class_one_y = [row[1] for row in classified_data_points if row[2] == 0]
        class_two_x = [row[0] for row in classified_data_points if row[2] == 1]
        class_two_y = [row[1] for row in classified_data_points if row[2] == 1]
        plt.scatter(class_one_x, class_one_y, label='0', color="red", marker="o", s=30)
        plt.scatter(class_two_x, class_two_y, label='1', color="blue", marker="o", s=30)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('Kernel Density Estimator Class Boundaries')
        plt.show()
