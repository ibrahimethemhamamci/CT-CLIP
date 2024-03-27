import numpy as np
import torch
import tqdm
import pandas as pd
from numpy.linalg import norm


def find_top_k_indices(values,k):
    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # Extract the indices of the top 50 values
    top_k_indices = [index for index, value in sorted_values_with_indices[:k]]

    return top_k_indices


def calc_similarity(arr1,arr2):
    oneandone=0
    oneorzero=0
    zeroandzero=0
    for k in range(len(arr1)):
        if arr1[k] == 0 and arr2[k] == 0:
            zeroandzero = zeroandzero +1

        if arr1[k] == 1 and arr2[k] == 1:
            oneandone = oneandone +1
        if arr1[k] == 0 and arr2[k] == 1:
            oneorzero = oneorzero +1
        if arr1[k] == 1 and arr2[k] == 0:
            oneorzero = oneorzero +1

    return ( (oneandone) / (oneandone + oneorzero))

data_folder = "path_to_latents_folder/"

file_path = data_folder + 'accessions.txt'

# Initialize an empty list to store the file contents
accs = []

# Open the file in read mode
with open(file_path, 'r') as file:
    # Read each line from the file and append it to the list
    for line in file:
        # Remove leading and trailing whitespace (e.g., newline characters)
        cleaned_line = line.strip()
        # Append the cleaned line to the list
        accs.append(cleaned_line)

# Now, file_contents contains the contents of the file as a list
df = pd.read_csv("path_to_validation_labels.csv")


image_data= np.load(data_folder + "image_latents.npz")["data"][:,0,:]


print(image_data.shape)
ratios_external=[]
image_data_for_second=[]
accs_for_second=[]
for k in tqdm.tqdm(range(image_data.shape[0])):
    acc_second=accs[k]
    #acc_second_last = acc_second[-1]
    #acc_second = acc_second[:-1]
    #acc_second = acc_second + "_" + acc_second_last + "_1.nii.gz"
    row_second = df[df['AccessionNo'] == acc_second]
    num_path=np.sum(row_second.iloc[:,1:].values[0])
    if num_path != 0:
        image_data_for_second.append(image_data[k])
        accs_for_second.append(accs[k])
image_data_for_second = np.array(image_data_for_second)
print(image_data_for_second.shape)


k_list = [1,5,10,50]
list_outs = []
for return_n in k_list:
    for i in tqdm.tqdm(range(image_data.shape[0])):
        first = image_data[i]
        size = (512)
        #first=torch.rand(size)
        acc_first = accs[i]
        #acc_first_last = acc_first[-1]
        #acc_first = acc_first[:-1]
        #acc_first = acc_first + "_" + acc_first_last + "_1.nii.gz"
        row_first = df[df['AccessionNo'] == acc_first]
        row_first = row_first.iloc[:,1:].values[0]

        crosses = []
        ratios_internal = []

        for k in range(image_data_for_second.shape[0]):
            second = image_data_for_second[k]
            size = (512)
            #second=torch.rand(size)
            #acc_second=accs[k]
            #row_second = df[df['AccessionNo'] == acc_second]

            #row_second = row_second.iloc[:,1:].values[0]

            #print(np.sum(row_first == row_second))
            # Calculate the dot product of the two vectors
            dot_product = np.dot(first, second)

            # Calculate the magnitude (Euclidean norm) of each vector
            magnitude_vector1 = norm(first)
            magnitude_vector2 = norm(second)

            # Calculate the cosine similarity
            cross = dot_product / (magnitude_vector1 * magnitude_vector2)
            crosses.append(cross)
        top_k_indices = find_top_k_indices(crosses,return_n)
        for index in top_k_indices:
            acc_second=accs_for_second[index]
            #acc_second_last = acc_second[-1]
            #acc_second = acc_second[:-1]
            #acc_second = acc_second + "_" + acc_second_last + "_1.nii.gz"
            
            row_second = df[df['AccessionNo'] == acc_second]

            row_second = row_second.iloc[:,1:].values[0]
            #ratio = np.sum(row_first == row_second) / 18
            ratio = calc_similarity(row_first,row_second)
            ratios_internal.append(ratio)
        #print(np.mean(np.array(ratios_internal)))
        ratios_external.append(np.mean(np.array(ratios_internal)))
    list_outs.append(str(np.mean(np.array(ratios_external))))

file_path = data_folder + "i2i_real.txt"

# Open the file for writing (you can also use "a" to append if the file already exists)
with open(file_path, "w") as file:
    # Write each string from the list to the file
    for string in list_outs:
        file.write(string + "\n")

# File has been written, close it
file.close()
