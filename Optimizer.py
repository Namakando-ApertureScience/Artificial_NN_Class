from Constructor import *
import math as m
import sys


class Network_optimizer(Network_constructor):

    def __init__(self, approximation_range, input_output_neuron_number, depth_width, weight_bound,
                 input_output_steepness, learning_rate_adam, rho1=0.9, rho2=0.999):

        super().__init__(approximation_range, input_output_neuron_number, depth_width, weight_bound,
                         input_output_steepness)

        # Recursion limit
        sys.setrecursionlimit(((depth_width[0] * depth_width[1]) + 100) ** 2)

        # Error vector
        self.error_vector = np.array([])

        # Important matrices for resilient propagation
        self.identity = np.full((depth_width[1], depth_width[1] + 1), 1.0)
        self.learning_rate_matrix = np.full((depth_width[0], depth_width[1], depth_width[1] + 1), 0.1)

        ################################################################################################################
        # Important constants

        # For Adam
        self.learning_rate_adam = learning_rate_adam

        # For resilient propagation
        self.learning_rate_up = 0
        self.learning_rate_down = 0

        self.rho1 = rho1
        self.rho2 = rho2

        # For Adam
        self.epsilon = 1e-10

        ################################################################################################################
        # For Adam

        # Time step
        self.time_step = 1
        self.time_step_bool = True

        # Moments
        self.A = np.zeros((self.depth_width[0], self.depth_width[1], self.depth_width[1] + 1))
        self.F = np.zeros((self.depth_width[0], self.depth_width[1], self.depth_width[1] + 1))

    ####################################################################################################################
    ####################################################################################################################
    # Gradient computation

    def Gradient(self, optimizer, learning_rate_BP, layer_number=0):

        if layer_number == self.depth_width[0] - 1:
            delta_vector = (self.approximation_range[1] - self.approximation_range[0]) * derivative_sigmoid(
                self.weighted_sums_matrix[layer_number],
                self.steepness_values[layer_number]) * self.error_vector

            (self.Backpropagation_prop_update(learning_rate_BP, layer_number, delta_vector) if optimizer == "Backpropagation"
             else (self.Resilient_prop_update(layer_number, delta_vector) if optimizer == "Resilient_Backpropagation"
                   else (self.Adam_update(layer_number, delta_vector) if optimizer == "Adam"
                         else print("Error !!!"))))

            return delta_vector

        delta_vector = (self.approximation_range[1] - self.approximation_range[0]) * derivative_sigmoid(
            self.weighted_sums_matrix[layer_number],
            self.steepness_values[layer_number]) * np.matmul(
            np.delete(self.neural_network[layer_number + 1].transpose(),
                      self.depth_width[1], axis=0), self.Gradient(optimizer, learning_rate_BP, layer_number + 1))

        (self.Backpropagation_prop_update(learning_rate_BP, layer_number, delta_vector) if optimizer == "Backpropagation"
         else (self.Resilient_prop_update(layer_number, delta_vector) if optimizer == "Resilient_Backpropagation"
               else (self.Adam_update(layer_number, delta_vector) if optimizer == "Adam"
                     else print("Error !!!"))))

        return delta_vector

    ####################################################################################################################
    ####################################################################################################################
    # Update

    def Updater(self, optimizer, input_vector, desired_output_vector, learning_rate_BP=0.1, learning_rate_up=1.05, learning_rate_down=0.5):

        self.error_vector = np.append((desired_output_vector - self.computation(input_vector)),
                                      np.zeros(len(self.output_deletion_vector)))

        self.learning_rate_up = learning_rate_up
        self.learning_rate_down = learning_rate_down

        self.Gradient(optimizer, learning_rate_BP)

        if optimizer == "Adam" and self.time_step_bool:
            self.time_step += 1

    ####################################################################################################################
    ####################################################################################################################
    # Backpropagation

    def Backpropagation_prop_update(self, learning_rate_BP, layer_number, delta_vector):
        self.neural_network[layer_number] = self.neural_network[layer_number] + learning_rate_BP * np.matmul(
            np.array([delta_vector]).transpose(), np.array([np.append(sigmoid(
                self.weighted_sums_matrix[layer_number - 1], self.steepness_values[layer_number - 1]), 1)]))

    ####################################################################################################################
    ####################################################################################################################
    # Resilient Backpropagation

    def Resilient_prop_update(self, layer_number, delta_vector):

        gradient = - np.matmul(
            np.array([delta_vector]).transpose(), np.array([np.append(sigmoid(
                self.weighted_sums_matrix[layer_number - 1], self.steepness_values[layer_number - 1]), 1)]))

        self.learning_rate_matrix[layer_number] = self.learning_rate_matrix[layer_number] * ((np.sign(gradient *
                                                                    self.gradients[layer_number]) *
                                                                (self.learning_rate_up - self.learning_rate_down) +
                                                                abs(np.sign(gradient * self.gradients[layer_number])) *
                                                                (self.learning_rate_up + self.learning_rate_down)) / 2 + \
                                                               abs(self.identity - abs(
                                                                   np.sign(gradient * self.gradients[layer_number]))))

        self.gradients[layer_number] = gradient

        self.neural_network[layer_number] = self.neural_network[layer_number] - \
                                            self.learning_rate_matrix[layer_number] * np.sign(gradient)

    ####################################################################################################################
    ####################################################################################################################
    # Adam and Adam-hybrid update

    def Adam_update(self, layer_number, delta_vector):

        if self.time_step_bool:
            bias_correction = m.sqrt(1 - self.rho2 ** self.time_step) / (1 - self.rho1 ** self.time_step)
            if abs(bias_correction - 1) < self.epsilon:
                self.time_step_bool = False
        else:
            bias_correction = 1

        grad = (np.matmul(np.array([delta_vector]).transpose(), np.array([np.append(sigmoid(
            self.weighted_sums_matrix[layer_number - 1], self.steepness_values[layer_number - 1]), 1)])))

        self.A[layer_number] = self.rho2 * self.A[layer_number] + (1 - self.rho2) * grad ** 2
        self.F[layer_number] = self.rho1 * self.F[layer_number] + (1 - self.rho1) * grad

        self.neural_network[layer_number] = (self.neural_network[layer_number] + (self.learning_rate_adam * bias_correction) *
                                             (self.F[layer_number] / ((self.A[layer_number] + self.epsilon) ** 0.5)))
