from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoTokenizer

MODEL_HF_ID = "google/siglip-so400m-patch14-384"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://dedup:dedup_local_dev@127.0.0.1:5432/dedup"
)
TOP_K = int(os.environ.get("TOP_K", "6"))
OUT_PATH = Path(os.environ.get("OUT_PATH", "/out/results.json"))

QUERIES = [
    "a photo of a dog",
    "a photo of a cat",
    "a beach with ocean waves",
    "snow covered mountains",
    "a red car",
    "a sunset sky",
    "flowers in bloom",
    "food on a plate",
    "a christmas tree with lights",
    "a baby",
    "a birthday cake with candles",
    "fireworks in the night sky",
    "a swimming pool",
    "a wedding ceremony",
    "a person playing a guitar",
    "a boat on the water",
    "a crowd of people at a concert",
    "fall leaves on trees",
    "a dog on a leash",
    "a person on a bicycle",
]

print(f"Loading SigLIP text encoder: {MODEL_HF_ID}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID)
model = AutoModel.from_pretrained(MODEL_HF_ID)
model.train(False)
print("Model loaded.", flush=True)

results: dict[str, list[tuple[str, float]]] = {}
with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)
    with conn.cursor() as setup:
        setup.execute("SET ivfflat.probes = 50")
    for q in QUERIES:
        inputs = tokenizer([q], padding="max_length", return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat[0].cpu().numpy().astype(np.float32)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sha256, 1 - (embedding <=> %s) AS sim
                FROM photo_embeddings
                WHERE media_type = 'photo'
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec, vec, TOP_K),
            )
            rows = cur.fetchall()
            results[q] = [(sha, float(sim)) for sha, sim in rows]
        if not results[q]:
            print(f"  {q!r:50s} -> NO ROWS", flush=True)
            continue
        top = results[q][0]
        print(f"  {q!r:50s} -> {top[0][:12]} ({top[1]:.3f})", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(results, indent=2))
print(f"Wrote {OUT_PATH}", flush=True)
