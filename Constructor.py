import numpy as np


def derivative_sigmoid(x, a):
    x = np.array(x, dtype=np.float128)
    return (np.exp(-x / a)) / (a * ((1 + np.exp(-x / a)) ** 2))


def sigmoid(x, a):
    x = np.array(x, dtype=np.float128)
    return 1 / (1 + np.exp(-x / a))


class Network_constructor:

    def __init__(self, approximation_range, input_output_neuron_number, depth_width, weight_bound,
                 input_output_steepness):

        # Network dimensions and approximation interval
        self.depth_width = depth_width
        self.approximation_range = approximation_range

        # Network input and output size correction
        self.input_length_diff = depth_width[1] - input_output_neuron_number[0]
        self.output_deletion_vector = (
                np.arange(depth_width[1] - input_output_neuron_number[1]) + input_output_neuron_number[1])

        # Sigmoid adjustment
        self.steepness_values = np.linspace(input_output_steepness[0], input_output_steepness[1], depth_width[0])

        # Important matrices
        self.network_copy = np.array([])
        self.weighted_sums_matrix = np.array([])
        self.gradients = np.full((depth_width[0], depth_width[1], depth_width[1] + 1), 0.0)

        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # Network construction

        # I. First layer construction
        self.neural_network = (weight_bound[1] - weight_bound[0]) * np.random.rand(depth_width[1] + 1) + weight_bound[0]

        for i in range(1, depth_width[1]):
            self.neural_network = np.vstack([self.neural_network, (weight_bound[1] - weight_bound[0]) *
                                             np.random.rand(depth_width[1] + 1) + weight_bound[0]])

        if depth_width[0] == 1:
            self.neural_network = np.array([self.neural_network])
            self.network_copy = self.neural_network.copy()
            return

        ################################################################################################################
        ################################################################################################################
        # II. Deeper layer construction

        # Second layer construction or making network compatible with vstack
        neural_network_layer = (weight_bound[1] - weight_bound[0]) * np.random.rand(depth_width[1] + 1) + weight_bound[0]

        for i in range(1, depth_width[1]):
            neural_network_layer = np.vstack([neural_network_layer, (weight_bound[1] - weight_bound[0]) *
                                              np.random.rand(depth_width[1] + 1) + weight_bound[0]])

        self.neural_network = np.stack([self.neural_network, neural_network_layer])

        ################################################################################################################

        # Third or greater layer construction
        for i in range(2, depth_width[0]):
            neural_network_layer = (weight_bound[1] - weight_bound[0]) * np.random.rand(depth_width[1] + 1) + weight_bound[0]

            for j in range(1, depth_width[1]):
                neural_network_layer = np.vstack([neural_network_layer, (weight_bound[1] - weight_bound[0]) *
                                                  np.random.rand(depth_width[1] + 1) + weight_bound[0]])

            self.neural_network = np.vstack((self.neural_network, np.array([neural_network_layer])))

        self.network_copy = self.neural_network.copy()

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # Network output computation

    def computation(self, input_vector, old_eval=False):

        # Network input correction
        input_vector = np.append(input_vector, np.zeros(self.input_length_diff))

        # Layer calculation
        if old_eval:
            self.weighted_sums_matrix = [np.matmul(self.network_copy[0], np.append(input_vector, 1))]

            for layer_number in range(1, self.depth_width[0]):
                self.weighted_sums_matrix = np.vstack([self.weighted_sums_matrix, np.matmul(
                    self.network_copy[layer_number], np.append(sigmoid(self.weighted_sums_matrix[layer_number - 1],
                                                                       self.steepness_values[layer_number - 1]), 1))])

        else:
            self.weighted_sums_matrix = [np.matmul(self.neural_network[0], np.append(input_vector, 1))]

            for layer_number in range(1, self.depth_width[0]):
                self.weighted_sums_matrix = np.vstack([self.weighted_sums_matrix, np.matmul(
                    self.neural_network[layer_number], np.append(sigmoid(self.weighted_sums_matrix[layer_number - 1],
                                                                         self.steepness_values[layer_number - 1]), 1))])

        ############################################################################
        # Network output

        network_output = sigmoid(self.weighted_sums_matrix[self.depth_width[0] - 1],
                       self.steepness_values[self.depth_width[0] - 1])

        ############################################################################
        # Converting standardised network output to approximation interval

        return np.delete((self.approximation_range[1] - self.approximation_range[0]) * \
                         network_output + self.approximation_range[0],
                         self.output_deletion_vector, axis=0)

    ####################################################################################################################
    # previous network iteration

    def copy_(self):
        self.network_copy = self.neural_network.copy()
