from Optimizer import *
import random as rd


# Standard error measure
def error_measure(Data_set_, file_bool=False, show_expected_output=False):
    print()

    if file_bool:
        file.write("\n")

    if show_expected_output:
        if file_bool:
            file.write("Approximation " + "Expected output" + "\n")

            for exa in Data_set_:
                file.write(str(network_1.computation(exa[0]).round(2))+' '+str(np.array(exa[1]).round(2))+"\n")

        else:
            print("Approximation", "Expected output")

            for exa in Data_set_:
                print(network_1.computation(exa[0]).round(2), np.array(exa[1]).round(2))

    else:
        for exa in Data_set_:
            if file_bool:
                file.write(str(network_1.computation(exa[0]).round(2))+"\n")
            else:
                print(network_1.computation(exa[0]).round(2))

    print()

    measure = 0
    for exa in Data_set_:
        measure += abs(network_1.computation(exa[0]) - exa[1]).sum()

    if file_bool:
        file.write("Approximate error measure: " + str(round(measure, 2))+"\n")
    else:
        print("Approximate error measure: " + str(round(measure, 2)))
        input()

    return measure


# Do you want a file
file_bool_ = False
if file_bool_:
    file = open("Training_data.txt", 'a')

# Relevant parameters
optimizer = "Resilient_Backpropagation"
approximation_range = [-1, 4]
input_output_neuron_number = [4, 2]
depth_width = [5, 7]
weight_bound = [-5, 5]
input_output_steepness = [1, 1]
learning_rate = 0.001

# Repeat exposures for training
exposures = 5_000

# Data generator
input_range = [-1, 1]
output_range = [0, 3]
Number_of_datapoints = 10
Data_set = []

for i in range(Number_of_datapoints):

    input_vector = []
    for j in range(input_output_neuron_number[0]):
        value = rd.random() * (input_range[1] - input_range[0]) + input_range[0]
        input_vector.append(value)

    output_vector = []
    for k in range(input_output_neuron_number[1]):
        value = rd.random() * (output_range[1] - output_range[0]) + output_range[0]
        output_vector.append(value)

    Data_set.append([input_vector, output_vector])

# Neural network
network_1 = Network_optimizer(approximation_range,
                              input_output_neuron_number,
                              depth_width,
                              weight_bound,
                              input_output_steepness,
                              learning_rate)

# Test
if file_bool_:
    file.write("You're using the {} optimizer.".format(optimizer))
else:
    print("You're using the {} optimizer.".format(optimizer))

old_error = error_measure(Data_set, file_bool_)

for i in range(exposures):
    print("repetitions: " + str(i + 1), end='\r')
    for j in range(Number_of_datapoints):
        Data = rd.choice(Data_set)
        network_1.Updater(optimizer, Data[0], Data[1])
print()

new_error = error_measure(Data_set, file_bool_, True)

error_improvement_percentage = 100 * (1 - new_error / old_error)

if file_bool_:
    file.write("Approximate error improvement: " + str(round(error_improvement_percentage, 2)) + "%")
    file.close()
else:
    print("Approximate error improvement: " + str(round(error_improvement_percentage, 2)) + "%")
