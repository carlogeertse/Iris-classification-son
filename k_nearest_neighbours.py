import math


class KNearestNeighbours():
    """
    Implementation of the k-nearest neighbours algorithm
    """

    def __init__(self, train_data, k):
        """
        Initialises some parameters
        :param train_data: The training data that contains an already known class
        :param k: The amount of neighbours to check in the algorithm
        """
        self.train_data = train_data
        self.k = k
        # Prepare some room for saving the distances
        for row in self.train_data:
            row.append(0)

    def predict(self, target):
        """
        Predicts the class of the target object based on the ealier supplied training data
        :param target: The target object whose class needs to be predicted
        :return: The predicted class of the target object
        """
        # Calculate the distance from all known data points to the target point
        for row in self.train_data:
            row[-1] = distance(row[:-2], target)
        # Find the k nearest neighbours
        neighbours = []
        train_copy = self.train_data[:]
        for i in range(0, self.k):
            minimum = train_copy[0][-1]
            minindex = 0
            for j in range(1, len(train_copy)):
                if train_copy[j][-1] < minimum:
                    minimum = train_copy[j][-1]
                    minindex = j
            neighbours.append(train_copy.pop(minindex))
        # Count the occurrences of the classes among the neighbours
        classes = {}
        for neighbour in neighbours:
            if neighbour[-2] in classes:
                classes[neighbour[-2]] += 1
            else:
                classes[neighbour[-2]] = 1
        # Get the class that occurs the most among the neighbours
        classification = max(classes, key=classes.get)
        return classification


def distance(v, w):
    """
    Calculates the Euclidian distance between two vectors with a similar amount of dimensions
    :param v: The first vector
    :param w: The second vector
    :return: The Euclidian distance between the vectors
    """
    distance = 0
    for i in range(0, len(v)):
        distance += (v[i] - w[i]) ** 2
    return math.sqrt(distance)
