import csv
import numpy as np
import random
import math
import sys

def formatted_print(arr):
    mx = len(max((sub[0] for sub in arr), key=len))
    for row in arr:
        print(" ".join(["{:<{mx}}".format(ele, mx=mx) for ele in row]))

class Point:
    def dist(pt1, pt2):
        sum = 0
        for i in range(0,len(pt1)):
            sum += math.pow(pt1[i]-pt2[i],2)
        return math.sqrt(sum)

    def avg(point):
        sum = 0
        for val in point:
            sum += val
        return sum / len(point)

class Centroids:
    def __init__(self, size, observations):
        self.data = None
        self.indices = None
        self.rows = None
        self.cols = None
        self.sums = None
        self.sum_sizes = None
        self._observations = None
        self._header = None

        self._observations = observations
        self._header = self._observations.header
        self.rows = size
        self.cols = self._observations.cols

        self.init_centroids()

    """ constructor functions """
    def init_centroids(self):
        self.data = np.empty(shape=(self.rows, self.cols))
        self.indices = [-1 for i in range(0,self.rows)]

        rand_index = -1
        for i in range(0, self.rows):
            # TODO :    wrap the random calculation in an inf loop that terminates only if its index hasn't been used before
            #           (already did this, but checking if a new centroid (self.data[rand_index]) is in the list of calculated centroids (self.data[0:i])
            #           always returns true)
            rand_index = random.randint(0, self._observations.rows-1)

            # TODO : find out if '.copy' is redundant in this case
            self.data[i] = self._observations.data[rand_index,].copy()
            self.indices[i] = rand_index

        self.zero_centroid_sums()

    """ utility functions """
    def zero_centroid_sums(self):
        self.sums = np.zeros(shape=(self.rows,self.cols))
        self.sum_sizes = [0.0 for i in range(0,self.rows)]

    def closest_centroid(self, observation):
        index = -1
        # makes sure first iteration will always be greater than distance
        min_dist = sys.maxsize
        for i, centroid in enumerate(self.data):
            dist = Point.dist(observation, centroid)
            if dist < min_dist:
                min_dist = dist
                index = i
        return index

    def update(self):
        # new centroids
        new_data = np.empty(shape=(self.rows,self.cols))

        for i in range(self.rows):
            # if no observation assigned to this cluster
            if self.sum_sizes[i] == 0:
                # skip it (retains the centroid as it was)
                continue
            new_data[i] = self.sums[i] / self.sum_sizes[i]

        difference = self.calc_centroid_difference(new_data)

        self.data = new_data

        return difference

    def calc_centroid_difference(self, new_data):
        avg_distance = 0
        for i in range(0, self.rows):
            avg_distance += Point.dist(self.data[i], new_data[i])
        avg_distance /= self.rows
        return avg_distance

    def print(self):
        print("Centroids:")
        labeled_data = np.append([self._header], self.data, axis=0)
        formatted_print(labeled_data);
        sys.stdout.write("Initial indices: "); print(self.indices)
        print()

    def add_sum(self, centroid_index, observation):
        self.sums[centroid_index] += observation
        self.sum_sizes[centroid_index] += 1


class Observations:
    def __init__(self, data_file_name):
        self.data = None
        self.data_centroids = None
        self.header = None
        self.rows = None
        self.cols = None
        self._centroids = None

        self.init_data(data_file_name)

    """ constructor functions """
    # initializes Data object with data from csv file, stripping header
    def init_data(self, data_file_name):
        # pull in the data to a temp list
        with open(data_file_name) as csv_file:
            list_data = list(csv.reader(csv_file, delimiter=","))

        # temp rows and cols
        rows = len(list_data)
        cols = len(list_data[0])

        # separate header from list (easier indexing)
        header_row = list_data[0]
        self.header = header_row[1:cols]

        # initialize Data object
        self.data = np.empty(shape=(rows - 1, cols - 1))

        # store data (without ids and header) into Data object
        for i in range(1, rows):
            self.data[i - 1] = list_data[i][1:cols]

        # set updated rows
        self.rows = rows - 1
        self.cols = cols - 1

        # initialize data centroids
        self.data_centroids = [-1 for i in range(0, self.rows)]

    """ constructor functions (cont.) [more of a post-constructor initialization function] """
    # links centroids to observations object (since centroid object wasn't created until after creation of observations)
    def link_centroids(self, centroids):
        self._centroids = centroids

    """ utility functions """
    def get_col(self, index):
        return [val[index] for val in self.data]

    def print(self):
        print("Observations:")
        centroid_list = [[-1 for i in range(0, self.cols)] for j in range(0, self.rows)]
        labeled_data = np.append([self.header], self.data, axis=0)
        formatted_print(labeled_data)
        print()

    # converts cluster names such that...:
    # initial clustering:       [3, 3, 3, 0, 1]
    # normalized clustering:    [1, 1, 1, 2, 3]
    # for readability / data standardization
    def get_normalized_indices(self):
        new_indices = [-1 for i in range(0, len(self.data_centroids))]
        new_centroid_indices = [-1 for i in range(0, self._centroids.rows)]

        index = 0
        for i in range(0, len(self.data_centroids)):
            if new_centroid_indices[self.data_centroids[i]] == -1:
                # if not in here, add it, update reference
                new_centroid_indices[self.data_centroids[i]] = index+1
                index += 1

        for i in range(0, len(self.data_centroids)):
            new_indices[i] = new_centroid_indices[self.data_centroids[i]]

        return new_indices

class KMeans:
    def __init__(self, data_file_name, num_centroids):
        self.observations = None
        self.centroids = None

        self.observations = Observations(data_file_name)
        self.centroids = Centroids(num_centroids, self.observations)

        self.link_centroids_to_observations()

    """ constructor functions """
    def link_centroids_to_observations(self):
        self.observations.link_centroids(self.centroids)

    """ utility functions """
    def print(self):
        print("KMeans:")
        self.observations.print()
        self.centroids.print()
        print()

    """ runtime functions """
    # do some things
        # for each observation
            # sort each into closest centroid
        # update centroids
    def iterate(self):
        self.centroids.zero_centroid_sums()

        for i, observation in enumerate(self.observations.data):
            closest_centroid_index = self.centroids.closest_centroid(observation)
            self.observations.data_centroids[i] = closest_centroid_index
            self.centroids.add_sum(closest_centroid_index, observation)

        # update the centroids and get the difference from last iteration
        difference = self.centroids.update()
        return difference

    def run(self, tolerance, iterations):
        print("\n" + ("+" * 40) + "\n              KMeans Run\n" + ("+" * 40) + "\n")

        print("Starting centroids:")
        self.centroids.print()

        difference = -1
        total_iterations = -1

        # do some rounds
        for i in range(0, iterations):
            difference = self.iterate()
            total_iterations += 1
            print("Difference: " + str(difference))
            self.centroids.print()
            if difference <= tolerance:
                print("\n" + ("*"*30) + " DONE " + ("*"*30) + "\n\n")
                break

        print("-KMeans Results: (last centroid difference = " + str(difference) + ")" + "\n")
        print("-Iterations: " + str(total_iterations) + "\n")
        sys.stdout.write("-Final "); self.centroids.print();
        print("-Cluster indices: "); print(self.observations.data_centroids); print()
        print("-Norm.d cluster indices (more readable): "); print(self.observations.get_normalized_indices())

        print("\n" + ("+"*40) + "\n              KMeans Run\n" + ("+"*40))

def main():
    tolerance = .01
    iterations = 1000

    k_means = KMeans("Data/IrisData.csv", 4)
    k_means.run(tolerance, iterations)

main()