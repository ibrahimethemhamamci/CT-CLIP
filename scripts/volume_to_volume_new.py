import numpy as np
import torch
import tqdm
import pandas as pd
import os
from numpy.linalg import norm
import tqdm

def find_top_k_indices(values, k):
    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    # Extract the indices of the top 50 values
    top_k_indices = [index for index, value in sorted_values_with_indices[:k]]
    return top_k_indices

def calc_similarity(arr1, arr2):
    oneandone = 0
    oneorzero = 0
    zeroandzero = 0
    for k in range(len(arr1)):
        if arr1[k] == 0 and arr2[k] == 0:
            zeroandzero += 1
        if arr1[k] == 1 and arr2[k] == 1:
            oneandone += 1
        if arr1[k] == 0 and arr2[k] == 1:
            oneorzero += 1
        if arr1[k] == 1 and arr2[k] == 0:
            oneorzero += 1

    return (oneandone / (oneandone + oneorzero))

data_folder = "/path_to_valid_latents_folder/image/"

# Scan the folder for .npz files
npz_files = [f for f in tqdm.tqdm(os.listdir(data_folder)) if f.endswith('.npz')]

# Initialize lists to store loaded data and accession numbers
image_data_list = []
accs = []

# Load each .npz file and use the filename (without extension) as the accession number
for npz_file in tqdm.tqdm(npz_files):
    file_path = os.path.join(data_folder, npz_file)
    image_data = np.load(file_path)["arr"][0]
    print(image_data.shape)
    image_data_list.append(image_data)
    accs.append(npz_file.replace("npz","nii.gz"))  # Use the filename without the extension as the accession number

# Concatenate all loaded image data
image_data = np.array(image_data_list)
print(image_data.shape)

# Load the validation labels
df = pd.read_csv("path_to_valid_predicted_labels.csv")

ratios_external = []
image_data_for_second = []
accs_for_second = []

# Filter the image data based on the condition in the validation labels
for k in tqdm.tqdm(range(image_data.shape[0])):
    acc_second = accs[k]
    row_second = df[df['VolumeName'] == acc_second]
    num_path = np.sum(row_second.iloc[:, 1:].values[0])
    if num_path != 0:
        image_data_for_second.append(image_data[k])
        accs_for_second.append(accs[k])

image_data_for_second = np.array(image_data_for_second)
print(image_data_for_second.shape)

k_list = [1, 5, 10, 50]
list_outs = []

# Calculate the similarity for each image in the dataset
for return_n in k_list:
    for i in tqdm.tqdm(range(image_data.shape[0])):
        first = image_data[i]
        acc_first = accs[i]
        row_first = df[df['VolumeName'] == acc_first]
        row_first = row_first.iloc[:, 1:].values[0]

        crosses = []
        ratios_internal = []

        for k in range(image_data_for_second.shape[0]):
            second = image_data_for_second[k]

            # Calculate the cosine similarity
            dot_product = np.dot(first, second)
            magnitude_vector1 = norm(first)
            magnitude_vector2 = norm(second)
            cross = dot_product / (magnitude_vector1 * magnitude_vector2)
            crosses.append(cross)

        top_k_indices = find_top_k_indices(crosses, return_n)
        for index in top_k_indices:
            acc_second = accs_for_second[index]
            row_second = df[df['VolumeName'] == acc_second]
            row_second = row_second.iloc[:, 1:].values[0]
            ratio = calc_similarity(row_first, row_second)
            ratios_internal.append(ratio)

        ratios_external.append(np.mean(np.array(ratios_internal)))
    list_outs.append(str(np.mean(np.array(ratios_external))))

# Write the output to a file
output_file_path = data_folder.replace("img_latents/", "") + "i2i_real.txt"

with open(output_file_path, "w") as file:
    for string in list_outs:
        file.write(string + "\n")