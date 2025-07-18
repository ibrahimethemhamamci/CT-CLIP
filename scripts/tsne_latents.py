import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os
import tqdm

def map_accessions_to_labels(accession, df):
    accession = accession.replace(".npz", ".nii.gz")
    row = df[df['VolumeName'] == accession]
    if not row.empty:
        try:
            return row.iloc[0, 1:]  # Return all label values as a numpy array
        except:
            return 0
    else:
        print(f"Label not found for {accession}")
        return np.zeros(df.shape[1] - 1)  # Return an array of zeros if no label found

def process_file(file_name, directory, df):
    if file_name.endswith(".npz"):
        file_path = os.path.join(directory, file_name)
        data = np.load(file_path, mmap_mode='r')['arr'][0]
        labels = map_accessions_to_labels(file_name, df)
        return data, labels
    return None, None

def load_latents_and_labels(directory, df):
    files = [f for f in os.listdir(directory) if f.endswith(".npz")]

    latents = []
    labels = []
    for file_name in tqdm.tqdm(files):
        data, label = process_file(file_name, directory, df)
        if data is not None:
            latents.append(data)
            labels.append(label)

    latents = np.vstack(latents)
    label_dict = {f: l for f, l in zip(files, labels)}

    return latents, label_dict

def tsne_projection(data, n_components=2, perplexity=30, n_iter=300):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=41)
    embedding = tsne.fit_transform(data)
    return embedding

def plot_tsne(embedding, labels, k, concat_dict):
    unique_labels = np.unique(labels)
    color_list = ["#000000", "#ff0066", "#117f80", "#ab66ff", "#66ccfc", "#FF7F50"]
    color_list_r = list(reversed(color_list))
    print(color_list_r)
    #color_list = ["#000000", "#ff0066", "#117f80"]
    annots = ["Others", f"Class {k + 1}", "4-6 Pathologies", "7-9 Pathologies", "10-12 Pathologies", ">13 Pathologies"]
    names_save = []
    for i, label in enumerate(reversed(unique_labels)):
        idx = np.where(labels == label)
        plt.scatter(embedding[idx, 0], embedding[idx, 1], s=1, alpha=0.8, color=color_list_r[i], label=f'{annots[i]}')
        if label == 0 or label== 1:
            for id in idx[0]:
                print(id)
                print("labeltrue")
                if embedding[id, 0] > 4:
                    if embedding[id,1] < 1:
                        keys_dict = list(concat_dict.keys())
                        names_save.append(keys_dict[id].replace(".npz",".nii.gz"))
                        print(keys_dict[id])

    pathologies = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
                       'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
                       'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                       'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
                       'Interlobular septal thickening']
    name = pathologies[k]

    plt.title(f"t-SNE (Image Latents)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    #plt.gca().invert_yaxis()
    plt.savefig(f"new_image_latents.png", dpi=600)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    latent_directory_train = "./path_to_latents/train/text_or_image"  #TODO: Directory containing train .npz files
    latent_directory_valid = "./path_to_latents/valid/text_or_image"  #TODO: Directory containing validation .npz files
    train_csv_path = "path_to_train_predicted_labels.csv"  #TODO: Path to train labels CSV
    validation_csv_path = "path_to_valid_predicted_labels.csv" #TODO: Path to validation labels CSV
    train_df = pd.read_csv(train_csv_path)
    validation_df = pd.read_csv(validation_csv_path)

    validation_latents, validation_label_dict = load_latents_and_labels(latent_directory_valid, validation_df)
    # Cache latents and labels
    train_latents, train_label_dict = load_latents_and_labels(latent_directory_train, train_df)

    all_latents = np.vstack([train_latents, validation_latents])
    #all_latents = validation_latents
    embedding = tsne_projection(all_latents)  # Compute t-SNE embedding only once

    def categorize_pathologies(count):
        if count == 0:
            return 0 # No pathology
        #else:
        #    return 1

        elif 1 <= count <= 3:
            return 1  # 1-3 Pathologies
        elif 4 <= count <= 6:
            return 2  # 4-6 Pathologies
        elif 7 <= count <= 9:
            return 3  # 7-9 Pathologies
        elif 10 <= count <= 12:
            return 4  # 10-12 Pathologies
        else:
            return 5  # >13 Pathologies

    for i in range(1):
        train_labels_count = np.array([np.sum(train_label_dict[file_name]) for file_name in train_label_dict.keys()])
        validation_labels_count = np.array([np.sum(validation_label_dict[file_name]) for file_name in validation_label_dict.keys()])

        # Combine train and validation labels
        combined_labels_count = np.concatenate([train_labels_count, validation_labels_count])


        concat_dict = train_label_dict | validation_label_dict
        # Categorize the counts into different groups
        combined_labels = np.array([categorize_pathologies(count) for count in combined_labels_count])

        plot_tsne(embedding, combined_labels, i, concat_dict)