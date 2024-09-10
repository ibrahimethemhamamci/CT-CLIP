from ct_clip.scripts.volume_to_volume import main


data_folder = "path_to_latents_folder/"
validation_labels_path = "path_to_validation_labels.csv"


if __name__ == "__main__":
    main(data_folder, validation_labels_path)
