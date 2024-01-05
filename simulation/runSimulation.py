import numpy as np
from mnist import loadDataset
from outputInterface import computePerformance
from inputInterface import imgToSpikeTrain

import os

# Time step duration in milliseconds
dt = 0.1
# simulation = "./"
simulation = "./simulation"
# Spikes trains duration in milliseconds
trainDuration = 350

# Number of computation steps
computationSteps = int(trainDuration / dt)

# Normalization of the input pixels' values
inputIntensity = 2.

# Number of images after which the accuracy is evaluated
updateInterval = 50

# Network shape
N_layers = 1
N_neurons = [400]
N_inputs = 784

# NumPy default random generator.
rng = np.random.default_rng()

# Mnist test dataset
images = simulation + "/mnist/t10k-images-idx3-ubyte"
labels = simulation + "/mnist/t10k-labels-idx1-ubyte"

# File containing the label associated to each neuron in the output layer
assignmentsFile = simulation + "/networkParameters/assignments.npy"

outputCountersFilename = "outputCounters.txt"

def write_input_spikes():
    # Import dataset
    imgArray, _ = loadDataset(images, labels)

    with open("./inputSpikes.txt", 'w') as filePointer:

        for i in range(updateInterval + 1):
            print(f"Iteration {i}")
            spikesTrains = imgToSpikeTrain(imgArray[i], dt, computationSteps, inputIntensity, rng)
            for step in spikesTrains:
                filePointer.write(str(list(step.astype(int)))
                                  [1:-1].replace(",", "").replace(" ", ""))
                filePointer.write("\n")


def compure_accuracy():
    directory_path = simulation + '/configurations'

    # Ottieni il percorso assoluto della cartella

    directory_contents = os.listdir(directory_path)
    
    outputCounters = np.zeros(N_neurons[-1]).astype(int)

    # Load the assignments from file
    with open(assignmentsFile, 'rb') as fp:
        assignments = np.load(fp)  # len = 400

    # Import dataset
    _, labelsArray = loadDataset(images, labels)

    for output in directory_contents:

        k = 0
        i = 0
        j = 0

        # Initialize history of spikes
        countersEvolution = np.zeros((updateInterval, N_neurons[-1]))
        accuracies = "0"
        
        # print(output); 
        # print(accuracies); 

        with open(directory_path + "/" + output, "r") as filePointer:
            for line in filePointer:

                outputCounters[k] = int(line)
                j += 1
                k = j % 400

                if k == 0:
                    countersEvolution[i % updateInterval] = outputCounters

                    i += 1

                    accuracies = computePerformance(i, updateInterval, countersEvolution, labelsArray, assignments,
                                                    accuracies)

        print(output.replace("_", ",").replace(".txt", "") + "," + accuracies)

        with open(simulation + '/logs/log.txt', 'a') as file:
            file.write(output.replace("_", ",").replace(".txt", "") + "," + accuracies + "\n")

def compare_files(): 
    directory_path = simulation + '/configurations'

    file1 = "VMem_Transient_13_28.txt"
    file2 = "VMem_Transient_34_247.txt"
    # file2 = "VReset_Transient_9_24.txt"

    vec_file1 = np.zeros((updateInterval, N_neurons[-1])) 
    vec_file2 = np.zeros((updateInterval, N_neurons[-1])) 

    temp = np.zeros(N_neurons[-1])

    i=0
    j=0
    k=0
      
    with open(directory_path + "/" + file1, "r") as filePointer:
        for line in filePointer:

            temp[k] = int(line)
            j += 1
            k = j % 400

            if k == 0:
                vec_file1[i % updateInterval] = temp

                i += 1

    i=0; j=0; k=0; 
    
    with open(directory_path + "/" + file2, "r") as filePointer:
        for line in filePointer:

            temp[k] = int(line)
            j += 1
            k = j % 400

            if k == 0:
                vec_file2[i % updateInterval] = temp

                i += 1

    
    # Trova gli indici in cui gli array sono diversi
    rows,cols = np.where(vec_file1 != vec_file2)
    # Stampa gli indici delle differenze
    print("Indici delle differenze:")
    for row, col in zip(rows, cols):
        print(f"Riga {row}, Colonna {col}")
    
    are_equal = np.array_equal(vec_file1, vec_file2)

    print(f"Arrays are equal: {are_equal}")
    

compure_accuracy()
#compare_files()