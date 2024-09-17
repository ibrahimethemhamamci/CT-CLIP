import logging
import os
from contextlib import asynccontextmanager
from logging import getLogger

import huggingface_hub
import uvicorn
from fastapi import FastAPI, Response, status
from pydantic import BaseModel, Field

from ct_clip.ct_clip.latents import CTClipLatents
from ct_clip.scripts.run_generate_latents import init_default_model


class GenerateLatentsInput(BaseModel):
    text: str


class GenerateLatentsResponse(BaseModel):
    text: str = None
    vector: list[float] = Field(default_factory=list)
    dim: int = 0
    error: str = None


logging.basicConfig(level=logging.INFO)
logger = getLogger("app")


latents: CTClipLatents | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    # model_dir = os.environ.get("MODEL_DIR", "models")
    # model_dir = Path(model_dir)
    # model_dir.mkdir(parents=True, exist_ok=True)
    model_path = "models/CT_CLIP_zeroshot.pt"
    if not os.path.exists(model_path):
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        if not huggingface_token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN env is not set. Cannot download CT-CLIP."
            )

        # Download CT-CLIP model weights
        logger.info("Downloading CT-CLIP model from huggingface...")
        model_path = huggingface_hub.hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            repo_type="dataset",
            filename="models/CT_CLIP_zeroshot.pt",
            local_dir=".",
            token=huggingface_token,
        )
        logger.info("CT-CLIP model downloaded to %s.", model_path)
    else:
        logger.info("CT-CLIP model already exists at path %s.", model_path)

    # Load the ML model
    global latents
    logger.info("Loading CT-CLIP model.")
    latents = init_default_model(model_path)
    logger.info(f"Loaded CT-CLIP on device {latents.accelerator.device}")

    yield
    # Clean up the ML models and release the resources
    latents = None


app = FastAPI(lifespan=lifespan)


@app.get("/.well-known/live", response_class=Response)
async def live(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/.well-known/ready", response_class=Response)
async def ready(response: Response):
    if latents is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_204_NO_CONTENT


@app.post("/latents", response_model=GenerateLatentsResponse)
@app.post("/latents/", response_model=GenerateLatentsResponse)
def generate_latents(
    item: GenerateLatentsInput, response: Response
) -> GenerateLatentsResponse:
    try:
        logger.info("Generating latent vectors")
        output = latents.generate_latents(text=item.text)
        latent_vector = output.texts[0]
        return GenerateLatentsResponse(
            text=item.text,
            vector=latent_vector.flatten().tolist(),
            dim=len(latent_vector),
        )
    except Exception as e:
        logger.exception("Something went wrong while generating latent vectors.")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return GenerateLatentsResponse(error=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
