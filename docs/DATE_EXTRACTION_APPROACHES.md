# Date Extraction from Scanned Photos: Approaches & Analysis

## Problem Statement

~7,500 scanned 4x6 photos (ScanMyPhotos Discs 1-4, plus ~22 in other scan folders) lack EXIF
metadata. Many have camera-imprinted date stamps — typically orange/red/yellow digits in a
bottom corner — but the format, color, position, and legibility vary. The dates span ~1986-2010.

**Goals:**

1. Read the printed date stamp from each photo (OCR)
2. Resolve ambiguous dates (is `04 05 '99` April 5 or May 4?)
3. Group photos by visual similarity (scene, clothing, faces) to cross-validate and correct dates
4. Output: a confident date for each photo, with a confidence score

**Constraints:**

- No discrete GPU (AMD Ryzen, integrated Vega, 27 GB RAM, 12 cores)
- Time is not a concern — accuracy matters more than speed
- Tesseract 5.3.4 already installed
- Budget flexibility, but costs should be justified

---

## Photo Inventory

| Source | Count | Notes |
|--------|------:|-------|
| ScanMyPhotos Disc 1 | 1,775 | Older photos, some rotated sideways, some lack stamps |
| ScanMyPhotos Disc 2 | 2,040 | Mix of stamped and unstamped |
| ScanMyPhotos Disc 3 | 2,076 | Mix of stamped and unstamped |
| ScanMyPhotos Disc 4 | 1,576 | More recent photos (2000s), more consistent stamps |
| Other scan folders | ~22 | Miscellaneous |
| **Total** | **~7,489** | |

---

## Date Stamp Characteristics (Observed)

From sampling actual photos across the discs:

| Photo | Stamp Text | Position | Color | Format |
|-------|-----------|----------|-------|--------|
| Disc 2/00000003 | `10 3 '99` | Bottom-right | Orange | MM D 'YY |
| Disc 2/00000023 | `11 19 '99` | Bottom-right | Orange | MM DD 'YY |
| Disc 2/00000043 | `12 3 '99` | Bottom-right | Orange-red | MM D 'YY |
| Disc 4/00000005 | `4 18 '03` | Bottom-left | Orange | M DD 'YY |
| Disc 2/00000013 | *(none)* | — | — | No stamp |
| Disc 1/00000003 | *(none)* | — | — | No stamp |
| Disc 3/00000005 | *(none)* | — | — | No stamp |

**Key observations:**

- Stamps are warm-colored (orange, red-orange, amber) — consistent with consumer camera date imprints of the era
- Position is usually bottom-right or bottom-left
- Format is `M(M) D(D) 'YY` with spaces, sometimes `MM/DD/YY` with slashes
- Leading zeros sometimes present, sometimes not
- Some photos have no stamp at all (older photos, or stamp was disabled)
- Some photos are rotated 90/270 degrees — stamp may appear on the side
- The `'YY` two-digit year means 1986-2010 maps to `'86` through `'10`

**Ambiguity challenge:** A stamp like `3 4 '99` could be March 4 or April 3 (if the camera was set to DD/MM). American cameras of this era almost always used MM/DD, but it's not guaranteed. Dates where both numbers are <= 12 are ambiguous.

---

## Approach Comparison Matrix

Scores are 1-10 (10 = best). Cost is for processing all ~7,500 images.

| Approach | Accuracy | Ambiguity Handling | Cost | Speed | Complexity | Setup Effort | Overall |
|----------|:--------:|:------------------:|:----:|:-----:|:----------:|:------------:|:-------:|
| **A. Traditional CV + Tesseract** | 4 | 2 | 10 | 9 | 5 | 7 | 5.3 |
| **B. Cloud Vision LLM (full image)** | 9 | 8 | 6 | 7 | 3 | 9 | 7.5 |
| **C. Local Vision LLM (full image)** | 7 | 7 | 10 | 2 | 5 | 5 | 6.0 |
| **D. Hybrid: CV crop + Cloud LLM** | 8 | 7 | 9 | 8 | 6 | 7 | 7.5 |
| **E. Hybrid: CV crop + Local LLM** | 7 | 6 | 10 | 3 | 6 | 5 | 6.0 |
| **F. Multi-model ensemble** | 10 | 9 | 4 | 3 | 8 | 6 | 6.7 |

**Recommended pipeline:** Start with **D** (cheapest good option), fall back to **B** for
failures, use **grouping** (Section 8) to resolve ambiguities. This gives you 9/10 accuracy at
minimal cost.

---

## Approach A: Traditional CV + OCR (Tesseract/EasyOCR/PaddleOCR)

### How It Works

1. **Crop bottom strip** — take the bottom 15% of the image (where stamps live)
2. **Color filter** — isolate orange/red pixels using HSV thresholds
   - Orange in HSV: H=5-25, S=100-255, V=150-255
   - Red-orange: H=0-10, S=100-255, V=150-255
3. **Binarize** — convert the isolated pixels to black-on-white
4. **OCR** — run Tesseract with `--psm 7` (single line) and digit whitelist
5. **Parse** — regex to extract date pattern from OCR output

### OCR Engine Options

| Engine | Accuracy on Date Stamps | Speed | Notes |
|--------|:-----------------------:|:-----:|-------|
| Tesseract 5.3 | Low-Medium | Fast | Already installed. Struggles with colored text on photo backgrounds. Best with clean binarization. |
| EasyOCR | Medium | Medium | Python-native, handles colored text better. Uses a small neural net. ~200MB download. |
| PaddleOCR | Medium-High | Medium | Best open-source OCR for scene text. Handles rotation, skew, colored text well. ~300MB. |
| TrOCR (Microsoft) | High | Slow (CPU) | Transformer-based, excellent at handwriting and scene text. ~1.5GB model. Very slow on CPU. |

### Concrete Implementation

```python
import cv2
import numpy as np
import pytesseract

def extract_date_stamp(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Try both bottom-right and bottom-left quarters
    crops = [
        img[int(h*0.85):, int(w*0.5):],   # bottom-right
        img[int(h*0.85):, :int(w*0.5)],    # bottom-left
    ]

    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Orange/red mask
        mask1 = cv2.inRange(hsv, (0, 80, 120), (25, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 80, 120), (180, 255, 255))
        mask = mask1 | mask2

        # Check if we found enough orange pixels
        if cv2.countNonZero(mask) < 50:
            continue

        # Dilate to connect digit segments
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Invert for Tesseract (black text on white)
        binary = cv2.bitwise_not(mask)

        text = pytesseract.image_to_string(
            binary,
            config='--psm 7 -c tessedit_char_whitelist=0123456789/. \''
        )
        return text.strip()

    return None  # No stamp found
```

### Strengths

- Completely free, runs locally
- Very fast (~0.1s per image)
- No API keys or network needed

### Weaknesses

- Orange text on complex photo backgrounds is hard to isolate cleanly
- Tesseract was designed for document text, not photo-embedded stamps
- Requires careful HSV tuning per batch — colors vary across cameras and scan quality
- Cannot understand context (season, clothing, events) to disambiguate dates
- Cannot handle rotated photos without additional rotation detection
- No ability to judge whether a detected date is plausible

### Cost

**$0.00** — Fully local. Tesseract already installed. OpenCV + numpy need `pip install`.

### Accuracy Estimate

- ~40-60% of stamps correctly read (dependent on binarization quality)
- High false-positive rate without date validation
- Cannot resolve any ambiguous dates

---

## Approach B: Cloud Vision LLM (Full Image)

### How It Works

Send the entire photo to a vision-capable LLM with a structured prompt asking it to:
1. Find and read the date stamp
2. Describe the scene (season, setting, event type)
3. Estimate the era from visual cues (clothing, technology, decor)
4. Flag whether the date is ambiguous (both fields <= 12)

### Model Options & Costs

All costs calculated for **7,500 images** with ~1,500 input tokens/image (image + prompt)
and ~150 output tokens (structured JSON response).

| Model | Input $/MTok | Output $/MTok | Image Tokens | Cost/Image | Total Cost | Quality |
|-------|:------------:|:-------------:|:------------:|:----------:|:----------:|:-------:|
| **Gemini 2.5 Flash** | $0.15 | $0.60 | ~1,300 | $0.00029 | **$2.17** | Good |
| **GPT-4o-mini** | $0.15 | $0.60 | ~765 (high detail) | $0.00020 | **$1.53** | Good |
| **Claude Haiku 4.5** | $1.00 | $5.00 | ~1,334 | $0.00208 | **$15.63** | Good |
| **GPT-4o** | $2.50 | $10.00 | ~765 | $0.00341 | **$25.56** | Very Good |
| **Claude Sonnet 4.6** | $3.00 | $15.00 | ~1,334 | $0.00625 | **$46.88** | Excellent |
| **Gemini 2.5 Pro** | $1.25 | $10.00 | ~1,300 | $0.00313 | **$23.44** | Excellent |
| **Claude Opus 4.6** | $15.00 | $75.00 | ~1,334 | $0.03125 | **$234.38** | Best |

*Note: Gemini 2.0 Flash is being deprecated June 2026. Use 2.5 Flash instead.*
*Note: Token counts are estimates — actual varies with image resolution. Scans at 300dpi
(1200x1800px) will use more tokens than the base estimates above. Multiply costs by ~1.5x
for safety margin.*

### Prompt Design

```
Analyze this scanned photo. Look for a camera date stamp — typically
orange, red, or amber digits printed in the bottom corner.

Return JSON:
{
  "stamp_found": true/false,
  "stamp_text_raw": "exactly what you see (e.g., '10 3 99')",
  "stamp_position": "bottom-right" | "bottom-left" | "other" | null,
  "date_parsed": "YYYY-MM-DD" or null,
  "date_ambiguous": true if both month and day are 1-12,
  "date_interpretation": "MM/DD/YY" or "DD/MM/YY" (your best guess),
  "confidence": 0.0-1.0,
  "era_estimate": "early 1990s" (from clothing, decor, technology),
  "season_guess": "fall" | "winter" | "spring" | "summer" | null,
  "scene_description": "two kids playing with blocks on kitchen floor",
  "notable_objects": ["cub scout uniform", "Little Tikes toy"],
  "rotation_needed": 0 | 90 | 180 | 270
}
```

### Strengths

- Highest accuracy — LLMs understand context, not just pixels
- Can read stamps even on busy/dark backgrounds
- Provides scene metadata useful for grouping and disambiguation
- Can estimate era from visual cues as a sanity check
- Handles rotated images naturally
- Structured output makes downstream processing easy
- Batch API support (Gemini, OpenAI) reduces cost further (~50% off)

### Weaknesses

- Costs money (though very cheap at the Gemini/GPT-4o-mini tier)
- Requires internet connectivity
- Rate limits may slow batch processing
- Occasional hallucination — LLM might "see" a date that isn't there
- Privacy: photos are sent to a third-party API

### Recommended Tier

**Primary: Gemini 2.5 Flash ($2-3 total)** — best cost/quality ratio for this task.
Use batch API for 50% discount.

**Verification pass on ambiguous/low-confidence: Claude Sonnet 4.6 or GPT-4o ($25-47 for
all, but likely only needed for ~1,000-2,000 photos)** — better reasoning for ambiguous cases.

### Batch API Savings

| Provider | Batch Discount | Effective Total Cost (7,500 imgs) |
|----------|:--------------:|:---------------------------------:|
| Gemini 2.5 Flash (batch) | 50% | **~$1.08** |
| OpenAI GPT-4o-mini (batch) | 50% | **~$0.77** |
| Anthropic Haiku (batch) | ~50% | **~$7.82** |

---

## Approach C: Local Vision LLM (Full Image)

### How It Works

Run an open-source vision-language model locally via Ollama, vLLM, or llama.cpp.
Same prompt as Approach B, but runs entirely on your hardware.

### Model Options (CPU-only, 27GB RAM)

| Model | Size (Q4) | RAM Needed | Speed (CPU) | OCR Quality | Scene Understanding |
|-------|:---------:|:----------:|:-----------:|:-----------:|:-------------------:|
| **Qwen2.5-VL 7B** | ~5 GB | ~8 GB | ~30-60s/img | Very Good | Good |
| **Gemma 3 4B** | ~3 GB | ~5 GB | ~15-30s/img | Good | Good |
| **LLaVA 1.6 7B** | ~5 GB | ~7 GB | ~30-60s/img | Medium | Good |
| **Phi-4-multimodal 14B** | ~9 GB | ~12 GB | ~60-120s/img | Very Good | Very Good |
| **Qwen3-VL 8B** | ~5 GB | ~8 GB | ~30-60s/img | Very Good | Very Good |
| **Llama 3.2 Vision 11B** | ~7 GB | ~10 GB | ~45-90s/img | Good | Good |
| **InternVL2.5 8B** | ~5 GB | ~8 GB | ~30-60s/img | Very Good | Good |

*Speeds are rough estimates for CPU inference with 12 cores. Actual performance depends
on image resolution and output length.*

### Recommended: Qwen2.5-VL 7B or Qwen3-VL 8B

Best open-source OCR accuracy as of 2026. Handles scene text (like date stamps) much
better than LLaVA variants. Native resolution input — no need to resize.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (~5GB download)
ollama pull qwen2.5-vl:7b

# Test on a single image
ollama run qwen2.5-vl:7b "Read the date stamp in this photo" --image /path/to/photo.jpg
```

### Processing Script

```python
import ollama
import json
import glob

def process_image(path):
    response = ollama.chat(
        model='qwen2.5-vl:7b',
        messages=[{
            'role': 'user',
            'content': PROMPT,  # same structured prompt as Approach B
            'images': [path]
        }],
        format='json'
    )
    return json.loads(response['message']['content'])

# Process all images (will take ~60-120 hours on CPU)
for img in sorted(glob.glob('/path/to/scans/*.jpg')):
    result = process_image(img)
    save_result(img, result)
```

### Time Estimates (CPU-only, 7,500 images)

| Model | Per Image | Total Time | Running 24/7 |
|-------|:---------:|:----------:|:------------:|
| Gemma 3 4B | ~20s | ~42 hours | 1.7 days |
| Qwen2.5-VL 7B | ~45s | ~94 hours | 3.9 days |
| Phi-4-multimodal 14B | ~90s | ~188 hours | 7.8 days |

### Strengths

- Completely free — no API costs
- Full privacy — photos never leave the machine
- Can run unattended for days
- Quality approaching cloud models for OCR tasks specifically

### Weaknesses

- Very slow on CPU (days, not hours)
- Lower accuracy than cloud models, especially for ambiguous cases
- Scene understanding and era estimation less reliable
- Needs ~8-12 GB RAM per model (feasible with 27 GB)
- Model output less structured/reliable — needs more parsing

### Cost

**$0.00** — electricity only (~$1-3 for multi-day CPU run at typical US rates)

---

## Approach D: Hybrid — CV Crop + Cloud LLM (Recommended Primary)

### How It Works

Combine the best of both worlds:
1. **OpenCV** isolates the bottom strip of the image and looks for orange-ish pixels
2. If orange pixels found → crop just the date stamp region (~100x30 pixels)
3. Send the **tiny crop** to a cloud LLM — dramatically fewer tokens than full image
4. If no orange pixels → flag as "no stamp" or send full image to LLM (fallback)

### Why This Is Smart

A 100x30 crop is ~32 tokens in most vision APIs, vs ~1,300+ tokens for a full image.
That's a **~40x reduction in token cost** for the date reading step.

### Token & Cost Comparison

| What's sent | Tokens | Cost/Image (Gemini 2.5 Flash) | Total (7,500) |
|-------------|:------:|:-----------------------------:|:-------------:|
| Full image | ~1,300 | $0.00029 | $2.17 |
| Cropped stamp only | ~50 | $0.00002 | $0.15 |
| Cropped stamp + text prompt | ~200 | $0.00005 | $0.38 |

### Implementation

```python
import cv2
import numpy as np

def find_and_crop_stamp(image_path):
    """Find orange date stamp, return cropped region or None."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Check all four edges (handles rotated photos)
    regions = {
        'bottom': img[int(h*0.85):, :],
        'top': img[:int(h*0.15), :],
        'right': img[:, int(w*0.85):],
        'left': img[:, :int(w*0.15)],
    }

    for position, region in regions.items():
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Broad orange/red/amber mask
        masks = [
            cv2.inRange(hsv, (0, 60, 100), (25, 255, 255)),    # orange-red
            cv2.inRange(hsv, (160, 60, 100), (180, 255, 255)),  # deep red
            cv2.inRange(hsv, (25, 60, 100), (35, 255, 255)),    # amber/yellow
        ]
        mask = masks[0] | masks[1] | masks[2]

        if cv2.countNonZero(mask) < 30:
            continue

        # Find bounding box of orange pixels
        coords = cv2.findNonZero(mask)
        x, y, rw, rh = cv2.boundingRect(coords)

        # Pad the crop
        pad = 15
        crop = region[max(0,y-pad):y+rh+pad, max(0,x-pad):x+rw+pad]

        return crop, position

    return None, None
```

Then send only the crop to the LLM:

```python
# Only ~50-200 tokens instead of ~1,300
response = gemini_model.generate(
    prompt="Read the date in this image. Return JSON: {date: 'MM/DD/YY', raw: '...'}",
    image=crop_bytes
)
```

### Strengths

- 10-40x cheaper than full-image cloud LLM
- Still leverages LLM intelligence for reading difficult stamps
- Fast preprocessing filters out no-stamp photos immediately
- Can pipeline: CV → LLM crop → full-image LLM fallback

### Weaknesses

- Two-step pipeline is more code to maintain
- CV step might miss stamps in unusual colors or positions
- Loses scene context from the crop — no era estimation or scene description
- Need a separate pass for scene analysis if grouping is desired

### Cost

**~$0.15-0.38 for crop-only via Gemini 2.5 Flash** (+ $2-3 for full-image fallback on
failures = **~$2.50-3.50 total**)

---

## Approach E: Hybrid — CV Crop + Local LLM

Same as Approach D but using a local model (Qwen2.5-VL, Gemma 3) for the crop reading step.
The tiny crop processes much faster than a full image on CPU.

### Time Estimates

| Model | Per Crop | Total (7,500 crops) | Speedup vs Full Image |
|-------|:--------:|:-------------------:|:---------------------:|
| Gemma 3 4B | ~3s | ~6.3 hours | 7x faster |
| Qwen2.5-VL 7B | ~8s | ~16.7 hours | 6x faster |

### Cost

**$0.00**

---

## Approach F: Multi-Model Ensemble

### How It Works

Run 2-3 approaches in parallel and combine results:
1. Tesseract on CV-cropped stamp (fast, free)
2. Local LLM on CV-cropped stamp (free, slower)
3. Cloud LLM on full image (cheap, best quality)

Take the majority vote. If all three disagree, flag for manual review.

### When This Makes Sense

Only if you need maximum accuracy and don't mind complexity. For ~7,500 photos,
the simpler approaches should handle 90%+ correctly, and grouping (Section 8) catches
most remaining errors.

### Cost

**~$2-4** (only the cloud pass costs money)

---

## Section 7: Date Disambiguation Strategy

This is the core intellectual challenge. A stamp reading `06 03 '98` — is that June 3 or
March 6?

### Rule-Based Disambiguation

| Rule | Logic | Confidence |
|------|-------|:----------:|
| **Day > 12** | If either number is 13-31, the other must be the month | 10/10 |
| **Year context** | `'86`-`'10` maps unambiguously to 1986-2010 | 10/10 |
| **American camera default** | US consumer cameras of this era defaulted to MM/DD/YY | 7/10 |
| **Format consistency within a roll** | If photo N-1 and N+1 are clearly MM/DD, photo N probably is too | 8/10 |
| **Season from scene** | Snow on ground + date `06 12 '98` → probably Dec 6, not June 12 | 8/10 |
| **Era from visual cues** | Fashion, technology, kids' ages can narrow the year | 6/10 |
| **Scan order** | Photos on the same disc are roughly chronological from the same era | 5/10 |

### Disambiguation Pipeline

```
1. Parse raw stamp text → extract two candidate dates
2. If one number > 12 → resolved (high confidence)
3. Check adjacent photos in scan order:
   a. If neighbors have unambiguous dates in MM/DD format → assume MM/DD
   b. If neighbors have dates that only make sense chronologically in one format → use that
4. Check scene for season cues (LLM scene description):
   a. Snow, heavy coats → winter months (Nov-Feb)
   b. Swimsuits, pools → summer months (Jun-Aug)
   c. Fall foliage → Sep-Nov
   d. Spring flowers → Mar-May
5. If still ambiguous → default to MM/DD/YY (American camera convention)
6. Flag remaining ambiguous dates for manual review
```

### Expected Ambiguity Rate

For dates from 1986-2010, both month and day are 1-12 in roughly **40-50%** of dates.
Of those, the American camera convention resolves most correctly. Season/context analysis
should catch another chunk. Estimated **5-10% of photos** will need manual disambiguation
after all automated methods.

---

## Section 8: Photo Grouping for Date Correction

This is where the magic happens — using visual similarity to catch and fix misread dates.

### 8.1 CLIP Embeddings for Scene Similarity

**What:** CLIP (Contrastive Language-Image Pretraining) generates a 512/768-dimensional
vector for each image. Cosine similarity between vectors indicates visual similarity.

**Why it works:** Photos from the same event/day tend to have similar backgrounds, lighting,
and subjects. A misread date will be an outlier in its group.

| Implementation | RAM | Speed (CPU) | Quality |
|---------------|:---:|:-----------:|:-------:|
| `clip-vit-base-patch32` (OpenAI) | ~1 GB | ~0.5s/img | Good |
| `clip-vit-large-patch14` (OpenAI) | ~2 GB | ~1.5s/img | Better |
| `SigLIP` (Google) | ~1 GB | ~0.5s/img | Better |
| `DFN-CLIP` (Apple) | ~2 GB | ~1.5s/img | Best |

**Cost: $0.00** — all run locally.

**Time for 7,500 images: ~1-3 hours on CPU.**

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

model = SentenceTransformer('clip-ViT-B-32')

# Generate embeddings for all images
embeddings = []
for img_path in all_images:
    emb = model.encode(Image.open(img_path))
    embeddings.append(emb)

embeddings = np.array(embeddings)

# Cluster visually similar photos
clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
clusters = clustering.fit_predict(embeddings)
```

### 8.2 Face Recognition for People Grouping

**What:** Detect and encode faces, then cluster by identity. Photos with the same people
in the same clothing are almost certainly from the same day/event.

| Implementation | RAM | Speed (CPU) | Quality |
|---------------|:---:|:-----------:|:-------:|
| `face_recognition` (dlib) | ~500 MB | ~2s/img | Good |
| `insightface` (ArcFace) | ~1 GB | ~1s/img | Better |
| `DeepFace` (wrapper) | ~1 GB | ~1.5s/img | Good (uses multiple backends) |

**Cost: $0.00**

**Key insight:** If photos A, B, C show the same people in the same outfits but A reads
as `03 05 '99` and B, C read as `05 03 '99`, the majority vote wins.

### 8.3 Color Histogram Similarity

**What:** Compare the overall color distribution of photos. Same-event photos often have
similar color palettes (same room, same outdoor lighting).

**Cost: $0.00, ~0.01s/image, no ML needed.**

```python
def color_histogram(image_path):
    img = cv2.imread(image_path)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    return cv2.normalize(hist, hist).flatten()

# Compare two images
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
```

### 8.4 Clothing Detection via CLIP Text Queries

**What:** Use CLIP's text-image matching to identify clothing:
- "person wearing blue cub scout uniform"
- "person wearing red shirt"
- "person wearing winter coat"

Same-clothing photos = same day. This is a free by-product if you're already computing
CLIP embeddings.

### 8.5 Grouping-Based Date Correction Algorithm

```
For each cluster of visually similar photos:
    1. Collect all extracted dates for photos in this cluster
    2. If all dates agree → high confidence, done
    3. If dates mostly agree with 1-2 outliers:
       a. Check if outlier is a plausible misread of the majority date
          (e.g., swapped month/day, off-by-one digit)
       b. If yes → correct the outlier to match majority
       c. If no → flag for manual review
    4. If dates are split (e.g., 3 say March, 3 say May):
       a. Use season/context analysis to break the tie
       b. If still tied → flag for manual review
    5. For photos with no stamp:
       a. If they cluster with stamped photos → inherit the cluster date
       b. If isolated → leave undated
```

---

## Section 9: Full Recommended Pipeline

### Stage 1: Preprocessing (free, ~2 hours)

1. **Rotation detection** — use OpenCV edge detection or a local model to identify and
   correct 90/180/270 rotations
2. **Stamp detection** — OpenCV color filtering to classify each photo as:
   - `HAS_STAMP` (orange pixels detected in edge region)
   - `NO_STAMP` (no orange pixels)
   - `UNCERTAIN` (some pixels but unclear)
3. **Stamp cropping** — extract the date stamp region for `HAS_STAMP` photos

### Stage 2: Date Extraction ($1-3, ~1-4 hours)

**Primary method: Gemini 2.5 Flash Batch API on crops**

- Send cropped stamp regions to Gemini 2.5 Flash via batch API
- Cost: ~$0.15-0.38 for crops, ~$1.08 for full-image fallback on failures
- Prompt asks for: raw text, parsed date, ambiguity flag, confidence

**Backup method: Full-image pass for failures**

- Photos where crop extraction failed or LLM confidence < 0.5
- Send full image to Gemini 2.5 Flash with scene analysis prompt
- Also handles `NO_STAMP` photos (LLM might spot stamps the CV missed)

**Optional verification: Second model pass on ambiguous dates**

- Send ambiguous dates (both numbers <= 12) to Claude Haiku or GPT-4o-mini
- Compare with Gemini result → take majority

### Stage 3: Grouping & Embedding (~3 hours, free)

1. Generate CLIP embeddings for all 7,500 photos
2. Run face detection + encoding on all photos
3. Cluster by visual similarity (DBSCAN on CLIP embeddings)
4. Sub-cluster by face identity within each visual cluster
5. Compute clothing similarity scores within face clusters

### Stage 4: Date Correction & Disambiguation (~minutes, programmatic)

1. Apply rule-based disambiguation (day > 12, adjacent photo consistency)
2. Apply cluster-based correction (majority vote within visual groups)
3. Apply season/context correction (LLM scene descriptions vs date)
4. Flag remaining ambiguous/conflicting dates for manual review
5. Generate confidence scores:
   - `HIGH` — unambiguous stamp + cluster agreement
   - `MEDIUM` — ambiguous stamp but cluster/context resolved
   - `LOW` — single photo, ambiguous, no context
   - `MANUAL` — conflicting signals, needs human review

### Stage 5: Output

- Write dates to EXIF using exiftool
- Generate `date_extraction_results.json` with full audit trail
- Generate `manual_review_needed.html` — a visual gallery of photos needing human input
  with side-by-side candidate dates and context

---

## Section 10: Cost Summary

| Component | Method | Cost | Time (CPU) |
|-----------|--------|-----:|:----------:|
| Stamp detection + cropping | OpenCV | $0.00 | ~30 min |
| Date OCR (crops) | Gemini 2.5 Flash batch | $0.15-0.38 | ~1 hour |
| Full-image fallback (~2,000 imgs) | Gemini 2.5 Flash batch | $0.58 | ~30 min |
| Ambiguity verification (~1,500 imgs) | GPT-4o-mini batch | $0.12 | ~20 min |
| Scene description (all) | Gemini 2.5 Flash batch | $1.08 | ~1 hour |
| CLIP embeddings | Local (clip-ViT-B-32) | $0.00 | ~1.5 hours |
| Face detection + encoding | Local (insightface) | $0.00 | ~2.5 hours |
| Clustering + correction | Local (Python) | $0.00 | ~5 min |
| **Total (recommended pipeline)** | | **$1.93-2.16** | **~7.5 hours** |

### Alternative: Fully Local Pipeline

| Component | Method | Cost | Time (CPU) |
|-----------|--------|-----:|:----------:|
| Stamp detection + cropping | OpenCV | $0.00 | ~30 min |
| Date OCR (crops) | Qwen2.5-VL 7B via Ollama | $0.00 | ~17 hours |
| Full-image fallback (~2,000 imgs) | Qwen2.5-VL 7B via Ollama | $0.00 | ~25 hours |
| Scene description (all) | Gemma 3 4B via Ollama | $0.00 | ~42 hours |
| CLIP embeddings | Local | $0.00 | ~1.5 hours |
| Face detection + encoding | Local | $0.00 | ~2.5 hours |
| **Total (fully local)** | | **$0.00** | **~88 hours (~3.7 days)** |

### Alternative: Maximum Accuracy (Cloud Heavy)

| Component | Method | Cost | Time |
|-----------|--------|-----:|:----:|
| Full-image analysis (all) | Claude Sonnet 4.6 | $46.88 | ~2 hours |
| Verification pass (ambiguous) | GPT-4o | $5.12 | ~30 min |
| CLIP + Face grouping | Local | $0.00 | ~4 hours |
| **Total (max accuracy)** | | **$52.00** | **~6.5 hours** |

---

## Section 11: Implementation Order

### Phase 1: Proof of Concept (1 session, ~100 images)

1. Install OpenCV + Pillow
2. Run stamp detection on 100 images from Disc 2 (known to have stamps)
3. Send 20 crops to Gemini 2.5 Flash — validate accuracy
4. Send same 20 as full images — compare results
5. Try Tesseract on same 20 crops — compare with LLM results
6. Install Ollama + Qwen2.5-VL 7B, test on same 20 crops
7. **Decision point:** which combination gives best results for this specific data?

### Phase 2: Full Date Extraction

1. Run CV stamp detection on all 7,500 images
2. Batch-send crops to chosen LLM
3. Full-image fallback for failures
4. Parse and validate all dates

### Phase 3: Grouping & Correction

1. Generate CLIP embeddings for all images
2. Run face detection/encoding
3. Cluster and cross-validate dates
4. Apply disambiguation rules
5. Generate manual review queue

### Phase 4: Integration

1. Write corrected dates to EXIF
2. Move into `organized/YYYY/YYYY-MM-DD/` structure
3. Archive results and audit trail

---

## Section 12: Tools & Dependencies

### Required

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.10+ | Pipeline orchestration | Already installed |
| OpenCV | Image processing, color filtering | `pip install opencv-python-headless` |
| Pillow | Image loading/saving | `pip install Pillow` |
| Tesseract 5.3 | Baseline OCR | Already installed |
| pytesseract | Python wrapper for Tesseract | `pip install pytesseract` |
| exiftool | EXIF read/write | `sudo apt install libimage-exiftool-perl` |

### For Cloud LLM Pipeline

| Tool | Purpose | Install |
|------|---------|---------|
| google-genai | Gemini API client | `pip install google-genai` |
| openai | OpenAI API client | `pip install openai` |
| anthropic | Claude API client | `pip install anthropic` |

### For Local LLM Pipeline

| Tool | Purpose | Install |
|------|---------|---------|
| Ollama | Local LLM runtime | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Qwen2.5-VL 7B | Vision-language model | `ollama pull qwen2.5-vl:7b` (~5 GB) |

### For Grouping/Clustering

| Tool | Purpose | Install |
|------|---------|---------|
| sentence-transformers | CLIP embeddings | `pip install sentence-transformers` |
| scikit-learn | DBSCAN clustering | `pip install scikit-learn` |
| insightface | Face recognition | `pip install insightface onnxruntime` |

---

## Section 13: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM hallucinates a date | Cross-validate with grouping; require stamp pixels to exist before trusting a read date |
| Tesseract misreads digits | Use LLM as primary, Tesseract only as tiebreaker |
| Color filter misses stamps | Try multiple HSV ranges; some stamps are yellow, red, or white — expand detection |
| Rotated photos | Run rotation detection first; check all four edges for stamps |
| Photos from non-US cameras (DD/MM) | Flag if user traveled internationally; cluster analysis helps |
| Scan quality varies | LLMs handle degraded text better than traditional OCR |
| Year ambiguity (does '02 mean 1902 or 2002?) | Constrain to 1986-2010 based on known collection era |
| Similar scenes on different days | Face clusters + clothing matching reduces false grouping |
| Processing interrupted | Save results incrementally; resume from last checkpoint |

---

## Section 14: Decision Matrix — Which Approach for You?

Given your setup (no GPU, 27GB RAM, time-flexible, ~7,500 photos):

| Priority | Best Choice | Why |
|----------|-------------|-----|
| **Cheapest + good accuracy** | Approach D (CV crop + Gemini Flash batch) | $2 total, ~7.5 hours |
| **Free + acceptable accuracy** | Approach E (CV crop + Qwen2.5-VL local) | $0, ~3.7 days |
| **Maximum accuracy** | Approach B (Claude Sonnet full image) + grouping | $47, ~6.5 hours |
| **Best balance overall** | D + grouping (recommended pipeline, Section 9) | $2, ~7.5 hours |

**My recommendation: Start with the proof-of-concept (Section 11, Phase 1) on 100 images.
This costs <$0.05 and takes ~1 hour. It will tell you exactly which approach works best
for YOUR specific photos before committing to a full run.**
