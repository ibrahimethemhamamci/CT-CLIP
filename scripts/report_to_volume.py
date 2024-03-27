import numpy as np
import torch
import tqdm

def find_top_k_indices(values,k):
    # Check if the list has at least 50 values
    if len(values) < 50:
        raise ValueError("The list must contain at least 50 values")

    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # Extract the indices of the top 50 values
    top_50_indices = [index for index, value in sorted_values_with_indices[:k]]

    return top_50_indices

data_folder = "path_to_latents_folder/"

image_data= np.load(data_folder+"image_latents.npz")["data"][:,0,:]
text_data = np.load(data_folder+"text_latents.npz")["data"][:,0,:]

print(image_data.shape)

list_texts = []
list_ks=[50]
for value in tqdm.tqdm(list_ks):
    num_is_in=0
    num_random=0

    for i in tqdm.tqdm(range(text_data.shape[0])):
        crosses = []
        crosses_rands=[]
        for k in range(image_data.shape[0]):
            text = torch.tensor(text_data[i])
            image = torch.tensor(image_data[k])

            cross = text @ image
            crosses.append(cross)

        top_k_indices = find_top_k_indices(crosses,value)
        if i in top_k_indices:
            num_is_in=num_is_in+1

        for k in range(image_data.shape[0]):
            size = (512)
            text =  torch.rand(size)
            image = torch.rand(size)

            crosses_rand= text @ image
            crosses_rands.append(crosses_rand)
        top_k_indices = find_top_k_indices(crosses_rands,value)
        if i in top_k_indices:
            num_random=num_random+1

    clip = num_is_in/text_data.shape[0]
    rand = num_random/text_data.shape[0]
    write_str = f"K={value}, clip = {clip}, rand= {rand}"
    list_texts.append(write_str)


file_path = data_folder + f"internal_accessions_t2i_{list_ks[0]}.txt"

# Open the file for writing (you can also use "a" to append if the file already exists)
with open(file_path, "w") as file:
    # Write each string from the list to the file
    for string in list_texts:
        file.write(string + "\n")

# File has been written, close it
file.close()
