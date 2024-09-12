import argparse

from transformers import BertTokenizer, BertModel

from ct_clip import CTCLIP, CTViT
from ct_clip.ct_clip.latents import CTClipLatents


def init_default_model(path_to_pretrained_model: str) -> CTClipLatents:
    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
    )
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=30,
        temporal_patch_size=15,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=2097152,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )

    return CTClipLatents(tokenizer, clip, path_to_pretrained_model)


def main(
    path_to_pretrained_model: str,
    texts: list[str] = None,
    image_paths: list[str] = None,
):
    if image_paths:
        raise NotImplementedError("Image latents are not yet supported")
    images = None

    if not texts and not image_paths:
        print("Nothing to do")
        return

    ctcliplatents = init_default_model(path_to_pretrained_model)

    print("Input text:", texts)
    print("Generating latents...")

    latents = ctcliplatents.generate_latents(texts=texts, images=images)
    print("Latents:", latents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-model",
        help="Path to pretrained CT-CLIP model file",
    )
    parser.add_argument(
        "--text", help="Input text string", action="append", dest="texts"
    )
    parser.add_argument(
        "--text-file",
        help="File of input text strings, one per line",
    )
    parser.add_argument(
        "--image-file",
        help="Path to input image file",
        action="append",
        dest="image_files",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    texts = args.texts or []
    if args.text_file:
        with open(args.text_file) as f:
            texts.extend(map(str.strip, f.readlines()))
    main(
        args.pretrained_model,
        texts,
        args.image_files,
    )


if __name__ == "__main__":
    main_cli()
