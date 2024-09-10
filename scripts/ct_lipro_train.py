from ct_clip.scripts import ct_lipro_train
from ct_clip.scripts.src.args import parse_arguments

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    # Start fine-tuning process
    ct_lipro_train.main(args)
