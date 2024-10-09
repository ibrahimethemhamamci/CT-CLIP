import numpy as np
import torch
import tqdm
import os

def find_top_k_indices(values, k):
    # Check if the list has at least 50 values
    if len(values) < k:
        raise ValueError(f"The list must contain at least {k} values")

    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # Extract the indices of the top k values
    top_k_indices = [index for index, value in sorted_values_with_indices[:k]]

    return top_k_indices

data_folder = "./path_to_save/"

# Scan the folder for image and text .npz files
image_npz_files = [f for f in os.listdir("/path_to_valid_latents_folder/image") if f.endswith('.npz')]
text_npz_files = [f for f in os.listdir("/path_to_valid_latents_folder/text") if f.endswith('.npz')]

# Initialize lists to store loaded data
image_data_list = []
text_data_list = []

# Load image and text .npz files
for npz_file in tqdm.tqdm(image_npz_files):
    file_path = os.path.join("/path_to_valid_latents_folder/image", npz_file)
    image_data = np.load(file_path)["arr"][0]
    image_data_list.append(image_data)

for npz_file in tqdm.tqdm(text_npz_files):
    file_path = os.path.join("/path_to_valid_latents_folder/text", npz_file)
    text_data = np.load(file_path)["arr"][0]
    text_data_list.append(text_data)

# Concatenate all loaded image and text data
image_data = np.array(image_data_list)
text_data = np.array(text_data_list)

print(image_data.shape)

list_texts = []
list_ks = [5,10,50,100]
for value in tqdm.tqdm(list_ks):
    num_is_in = 0
    num_random = 0

    for i in tqdm.tqdm(range(text_data.shape[0])):
        crosses = []
        crosses_rands = []
        for k in range(image_data.shape[0]):
            text = torch.tensor(text_data[i])
            image = torch.tensor(image_data[k])

            cross = text @ image
            crosses.append(cross)

        top_k_indices = find_top_k_indices(crosses, value)
        if i in top_k_indices:
            num_is_in += 1

        for k in range(image_data.shape[0]):
            size = (512)
            text = torch.rand(size)
            image = torch.rand(size)

            crosses_rand = text @ image
            crosses_rands.append(crosses_rand)
        top_k_indices = find_top_k_indices(crosses_rands, value)
        if i in top_k_indices:
            num_random += 1

    clip = num_is_in / text_data.shape[0]
    rand = num_random / text_data.shape[0]
    write_str = f"K={value}, clip = {clip}, rand= {rand}"
    list_texts.append(write_str)

output_file_path = data_folder + f"internal_accessions_t2i_{list_ks[0]}.txt"

# Open the file for writing (you can also use "a" to append if the file already exists)
with open(output_file_path, "w") as file:
    # Write each string from the list to the file
    for string in list_texts:
        file.write(string + "\n")

# File has been written, close it
file.close()