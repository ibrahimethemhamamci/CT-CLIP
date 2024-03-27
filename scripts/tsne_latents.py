import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np
from sklearn.manifold import TSNE  # Import TSNE from sklearn.manifold

def read_txt(file_path):
    with open(file_path, 'r') as f:
        accessions = f.readlines()
    return [line.strip() for line in accessions]

def map_accessions_to_labels(accessions, csv_path, i, isValid=False):
    # Load the csv file
    df = pd.read_csv(csv_path)

    # For each accession, find the corresponding row in the dataframe
    # and sum the labels. Store the results in a list.
    label_sums = []
    for acc in tqdm.tqdm(accessions):
        if isValid:
            acc_last = acc[-1]
            acc = acc[:-1]
            acc = acc + "_" + acc_last + "_1.nii.gz"
        row = df[df['AccessionNo'] == acc]

        if not row.empty:
            #label_sum = row.iloc[:, 1:].sum(axis=1).values[0]  # Assuming first column is AccessionNo
            label = row.iloc[:,i+1].values[0]
            #print(label)
            """
            if label_sum == 0:
                label = 1
            elif 4 > label_sum > 0:
                label = 0
            elif 7 > label_sum > 3:
                label = 0
            elif 10 > label_sum > 6:
                label = 0
            elif 13 > label_sum > 9:
                label = 0
            elif label_sum > 12:
                label = 0
            """
            label_sums.append(label)
        else:
            label_sums.append(0)  # Default to 0 if accession not found

    return np.array(label_sums)

def load_and_concatenate(train_path1, train_path2, validation_path):
    # Load the .npz files
    train_data1 = np.load(train_path1)
    train_data2 = np.load(train_path2)
    validation_data = np.load(validation_path)

    # Extract the latent values using the assumed key 'latent_values'
    train_latents1 = train_data1['data']
    train_latents2 = train_data2['data']
    validation_latents = validation_data['data']
    all_latents = np.vstack([train_latents1,train_latents2, validation_latents])
    all_latents = all_latents[:, 0, :]

    return all_latents

def tsne_projection(data, n_components=2, perplexity=30, n_iter=300):
    # Create a TSNE object with the desired parameters
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=1231)

    # Fit the TSNE model to your data and transform it
    embedding = tsne.fit_transform(data)
    return embedding

def plot_tsne(embedding, labels, k):
    unique_labels = np.unique(labels)
    color_list = ["#000000", "#ff0066", "#117f80", "#ab66ff", "#66ccfc", "#FF7F50"]
    annots = ["Others", f"Class {k + 1}", "4-6 Pathologies", "7-9 Pathologies", "10-12 Pathologies", ">13 Pathologies"]
    i = 0
    for label in tqdm.tqdm(unique_labels):
        idx = np.where(labels == label)
        plt.scatter(embedding[idx, 0], embedding[idx, 1], s=2, alpha=0.5, color=color_list[i], label=f'{annots[i]}')
        i = i + 1

    #name = "t-SNE"
    pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
    name = pathologies[k]

    plt.title(f"{name} (Image Latents)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(f"tsne_plots_test2/image_diseases/{name}_healthyornot_text.png", dpi=600)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    train_path_txt = "path_to_train_accessions.txt"
    validation_path_txt = "path_to_valid_accessions.txt"
    train_csv_path = "path_to_train_labels.csv"
    csv_path = "path_to_valid_labels.csv"

    train_accessions = read_txt(train_path_txt)
    validation_accessions = read_txt(validation_path_txt)

    for i in tqdm.tqdm(range(18)):
        train_labels_sum = map_accessions_to_labels(train_accessions, train_csv_path, i)
        validation_labels_sum = map_accessions_to_labels(validation_accessions, csv_path, i, isValid=True)

        combined_labels_sum = np.concatenate([train_labels_sum, validation_labels_sum])

        train_path1 = "path_to_train_image_latents.npz" ##could be done for text as well
        validation_path = "path_to_valid_image_latents.npz" ##could be done for text as well

        if i == 0:
            all_latents = load_and_concatenate(train_path1,train_path2, validation_path)
            embedding = tsne_projection(all_latents)

        plot_tsne(embedding, combined_labels_sum, i)
