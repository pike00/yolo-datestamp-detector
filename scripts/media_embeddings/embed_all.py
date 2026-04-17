from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from media_embeddings.db import MODEL_NAME, bulk_insert_embeddings, get_completed_stems
from media_embeddings.loader import extract_keyframes, open_image, scan_media_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "/home/will/photo_project/originals/media"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)
MODEL_HF_ID = "google/siglip-so400m-patch14-384"
IMAGE_BATCH = 64
VIDEO_BATCH = 16
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))

_num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
if _num_threads > 0:
    torch.set_num_threads(_num_threads)


def embed_batch(
    model: AutoModel,
    processor: AutoProcessor,
    images: list[Image.Image],
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding="max_length")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    features = out.pooler_output if hasattr(out, "pooler_output") else out
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def process_images(
    conn: psycopg.Connection,
    model: AutoModel,
    processor: AutoProcessor,
    image_paths: list[Path],
    done_stems: set[str],
) -> int:
    pending = [p for p in image_paths if p.stem not in done_stems]
    log.info(
        "Images: %d total, %d already done, %d to embed",
        len(image_paths), len(image_paths) - len(pending), len(pending),
    )
    processed = 0
    for i in tqdm(range(0, len(pending), IMAGE_BATCH), desc="images", unit="batch"):
        batch_paths = pending[i : i + IMAGE_BATCH]
        images, stems = [], []
        for path in batch_paths:
            try:
                images.append(open_image(path))
                stems.append(path.stem)
            except Exception as exc:
                log.warning("Skipping %s: %s", path.name, exc)
        if not images:
            continue
        vectors = embed_batch(model, processor, images)
        rows = [
            (stem, MODEL_NAME, vec.tolist(), "photo", None)
            for stem, vec in zip(stems, vectors)
        ]
        bulk_insert_embeddings(conn, rows)
        processed += len(rows)
    return processed


def process_videos(
    conn: psycopg.Connection,
    model: AutoModel,
    processor: AutoProcessor,
    video_paths: list[Path],
    done_stems: set[str],
) -> int:
    pending = [p for p in video_paths if p.stem not in done_stems]
    log.info(
        "Videos: %d total, %d already done, %d to embed",
        len(video_paths), len(video_paths) - len(pending), len(pending),
    )
    processed = 0
    for i in tqdm(range(0, len(pending), VIDEO_BATCH), desc="videos", unit="batch"):
        batch_paths = pending[i : i + VIDEO_BATCH]
        for path in batch_paths:
            try:
                frames = extract_keyframes(path, n=3)
                vectors = embed_batch(model, processor, frames)
                rows = [
                    (path.stem, MODEL_NAME, vec.tolist(), "video_keyframe", idx)
                    for idx, vec in enumerate(vectors)
                ]
                bulk_insert_embeddings(conn, rows)
                processed += len(frames)
            except Exception as exc:
                log.warning("Skipping video %s: %s", path.name, exc)
    return processed


def main() -> None:
    log.info("Loading SigLIP model: %s", MODEL_HF_ID)
    processor = AutoProcessor.from_pretrained(MODEL_HF_ID)
    model = AutoModel.from_pretrained(MODEL_HF_ID)
    model.eval()
    log.info("Model loaded. Scanning %s", MEDIA_DIR)

    image_paths, video_paths = scan_media_dir(MEDIA_DIR)
    log.info("Found %d images, %d videos", len(image_paths), len(video_paths))

    if SHARD_COUNT > 1:
        image_paths = image_paths[SHARD_INDEX::SHARD_COUNT]
        video_paths = video_paths[SHARD_INDEX::SHARD_COUNT]
        log.info(
            "Shard %d/%d: %d images, %d videos assigned to this worker",
            SHARD_INDEX, SHARD_COUNT, len(image_paths), len(video_paths),
        )

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        done_photos, done_videos = get_completed_stems(conn, MODEL_NAME)
        log.info(
            "Checkpoint: %d photos done, %d videos done",
            len(done_photos), len(done_videos),
        )
        n_images = process_images(conn, model, processor, image_paths, done_photos)
        n_videos = process_videos(conn, model, processor, video_paths, done_videos)

    log.info("Done. Embedded %d images, %d video frames.", n_images, n_videos)


if __name__ == "__main__":
    main()
