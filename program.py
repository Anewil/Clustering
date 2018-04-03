import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import math
import os
import shutil
import imageio
from sklearn import preprocessing
from scipy.spatial.distance import euclidean as distance

class Dataset:
    def __init__(self, file, nuber_of_neurons, usecols):
        self.data = np.loadtxt(file, delimiter=',', usecols=usecols)
        self.scale()
        self.number_of_neurons = nuber_of_neurons
        self.length = len(self.data)
        self.generate_neurons()

    def random_point(self):
        point = self.data[np.random.randint(self.length)]
        return point

    def generate_neurons(self):
        neurons = []
        for i in range(self.number_of_neurons):
            neurons.append(self.random_point())
        self.neurons = np.asarray(neurons)

    def scale(self):
        self.data = preprocessing.scale(self.data)

    def print_dataset(self):
        print(self.data)

    def count_groups(self):
        counter = np.zeros(self.neurons.shape[0])
        for row in self.data:
            minn = 0
            min = distance(row, self.neurons[0])
            for n in range(1, self.number_of_neurons):
                dist = distance(row, self.neurons[n])
                if dist < min:
                    min = dist
                    minn = n
            counter[minn] += 1
        print(counter)

    def sorted_neurons_by_distance(self, point):
        distances = []
        for i in range(self.number_of_neurons):
            distances.append(distance(self.neurons[i], point))
        distances = np.reshape(distances, (self.number_of_neurons, 1))
        self.neurons = np.hstack((self.neurons, distances))
        self.neurons = self.neurons[np.argsort(self.neurons[:, -1])]
        self.neurons = np.delete(self.neurons, -1, 1)
        return self.neurons

    def get_nearest_n(self, point):
        nearest = distance(point, self.neurons[0])
        nearestn = 0
        for n in range(self.number_of_neurons):
            dist = distance(point, self.neurons[n])
            if dist < nearest:
                nearest = dist
                nearestn = n
        return nearestn

    def get_nearest_neuron(self, point):
        return self.neurons[self.get_nearest_n(point)]

    def generate_plot(self, col1, col2, filename):
        plt.ion()
        plt.clf()
        colors = ['blue', 'yellow', 'green', 'purple', 'orange', 'black', 'brown']
        if not os.path.isdir('tmp'):
            os.makedirs('tmp')
        filename = str(filename).rjust(4, '0')
        self.neurons = self.sorted_neurons_by_distance(np.average(self.neurons))
        for row in self.data:
            n = self.get_nearest_n(row)
            plt.plot(row[col1], row[col2], marker='o', markersize=3, color=colors[n % 7])
        for neuron in self.neurons:
            plt.plot(neuron[col1], neuron[col2], marker='x', markersize=5, color='red')
        plt.show()
        plt.pause(0.2)
        filename = 'tmp/' + filename + '.png'
        plt.savefig(filename)

    def shuffle(self):
        np.random.shuffle(self.data)

    def create_animation(self, name):
        list = os.listdir('tmp')
        with imageio.get_writer(name, mode='I', duration=0.1) as writer:
            for filename in list:
                image = imageio.imread('tmp/' + filename)
                writer.append_data(image)
        writer.close()
        shutil.rmtree('tmp')


class NeuralGas:
    def __init__(self, dataset, lamb, epochs, lrate, generate_plot=False, col1=None, col2=None):
        lrate_i = 0.002
        for e in range(epochs):
            if generate_plot:
                dataset.generate_plot(col1, col2, e)
            dataset.shuffle()
            for row in dataset.data:
                dataset.neurons = dataset.sorted_neurons_by_distance(row)
                for n in range(dataset.number_of_neurons):
                    dataset.neurons[n] += lrate * pow(math.e, -n / lamb) * (row - dataset.neurons[n])
            if lrate - lrate_i > 0:
                lrate -= lrate_i
        dataset.create_animation('neuralgas.gif')
        dataset.count_groups()
        dataset.generate_neurons()


class Kohonen:
    def __init__(self, dataset, lamb, epochs, lrate, generate_plot=False, col1=None, col2=None):
        lrate_i = 0.002
        for e in range(epochs):
            if generate_plot:
                dataset.generate_plot(col1, col2, e)
            dataset.shuffle()
            for row in dataset.data:
                for n in range(dataset.number_of_neurons):
                    nearest = dataset.get_nearest_neuron(row)
                    dist = distance(dataset.neurons[n], nearest)
                    dataset.neurons[n] += lrate * self.gaussian_neighborhood(dist, lamb) * (row - dataset.neurons[n])
            if lrate - lrate_i > 0:
                lrate -= lrate_i
        dataset.create_animation('kohonen.gif')
        dataset.count_groups()
        dataset.generate_neurons()

    def gaussian_neighborhood(self, distance, lamb):
        return np.exp(-(pow(distance, 2) / 2.0 * pow(lamb, 2)))


class Kmeans:
    def __init__(self, dataset, generate_plot=False, col1=None, col2=None):
        self.i = 0
        self.q_error = self.quantization_error(dataset)
        self.neurons = dataset.neurons
        for i in range(10):
            dataset.generate_neurons()
            self.kmeans(dataset, generate_plot, col1, col2)
        dataset.neurons = self.neurons
        self.kmeans(dataset, generate_plot, col1, col2)
        dataset.create_animation('kmeans.gif')
        dataset.count_groups()
        dataset.generate_neurons()

    def kmeans(self, dataset, generate_plot=False, col1=None, col2=None):
        start_neurons = dataset.neurons
        while True:
            previous_neurons = dataset.neurons.copy()
            if generate_plot:
                dataset.generate_plot(col1, col2, self.i)
                self.i += 1
            for g in range(dataset.number_of_neurons):
                group = np.empty(shape=(0, len(dataset.data[0])))
                for r in range(dataset.length):
                    distances = np.zeros(shape=dataset.number_of_neurons)
                    for n in range(dataset.number_of_neurons):
                        distances[n] = distance(dataset.neurons[n], dataset.data[r])
                    index = np.where(distances == min(distances))
                    index = int(index[0][0])
                    if g == index:
                        group = np.vstack((group, dataset.data[r]))
                dataset.neurons[g] = np.mean(group, axis=0)
            if np.array_equal(dataset.neurons, previous_neurons):
                break
        error = self.quantization_error(dataset)
        if error < self.q_error:
            self.q_error = error
            self.neurons = start_neurons

    def quantization_error(self, dataset):
        sum = 0
        for row in dataset.data:
            neuron = dataset.get_nearest_neuron(row)
            sum += distance(row, neuron)
        return sum


dataset = Dataset('iris.data', 3, (0, 1, 2, 3))

neuralgas = NeuralGas(dataset, 0.33, 100, 0.1, True, 2, 3)
kohonen = Kohonen(dataset, 2, 100, 0.1, True, 2, 3)
kmeans = Kmeans(dataset, True, 2, 3)
