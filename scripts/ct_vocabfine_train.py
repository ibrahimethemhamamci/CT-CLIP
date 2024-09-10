from ct_clip.scripts.ct_vocabfine_train import main
from ct_clip.scripts.src.args import parse_arguments

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
