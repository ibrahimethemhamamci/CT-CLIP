import argparse

from ct_clip.scripts.run_zero_shot import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-model",
        help="Path to pretrained CT-CLIP model file",
    )
    parser.add_argument(
        "--preprocessed-validation-folder",
        help="Path to preprocessed validation folder",
    )
    parser.add_argument(
        "--validation-reports-csv",
        help="Path to validation reports csv",
    )
    parser.add_argument(
        "--validation-labels-csv",
        help="Path to validation labels csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.path_to_pretrained_model,
        args.preprocessed_validation_folder,
        args.validation_reports_csv,
        args.validation_labels_csv,
    )
