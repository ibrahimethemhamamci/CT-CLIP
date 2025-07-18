import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Number of iterations to print",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Number of iterations to save",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained CT-CLIP model path",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Data folder path",
    )
    parser.add_argument(
        "--reports-file",
        type=str,
        default=None,
        help="Reports file path",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        default=None,
        help="Meta file path",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Labels path",
    )


    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
