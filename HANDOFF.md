# Handoff: Scanned Photo Date Extraction Pipeline

**Generated**: 2026-04-03
**Status**: In Progress — proof of concept partially built, stamp detection needs rework

## Goal

Build a pipeline to extract camera-imprinted date stamps from ~7,500 scanned 4x6 photos (ScanMyPhotos Discs 1-4), resolve ambiguous dates (MM/DD vs DD/MM), and group photos by visual similarity to cross-validate and correct dates. Photos span ~1986-2010, have no EXIF data.

## Completed

- [x] Comprehensive strategy document with scored approaches: `DATE_EXTRACTION_APPROACHES.md`
- [x] Photo inventory: 4 discs (1775 + 2040 + 2076 + 1576 = ~7,467 files), plus ~22 in other scan folders
- [x] Sample analysis of date stamp characteristics (orange digits, varying position/format)
- [x] Python venv created at `.venv/` with OpenCV, Pillow, transformers, torch installed
- [x] First draft of `stamp_detect.py` — OpenCV-based stamp region detection + cropping
- [x] Ran stamp detection on 100 Disc 2 images — **produced crops but detection is broken** (see Failed Approaches)

## Not Yet Done

- [ ] Fix stamp detection to avoid false positives (current version catches any warm-colored object)
- [ ] Run TrOCR (Microsoft transformer OCR) on correctly-detected stamp crops
- [ ] Install Ollama + Qwen2.5-VL 7B and test on crops
- [ ] Compare TrOCR vs Tesseract vs local LLM accuracy on same crops
- [ ] Run corrected pipeline on full 100-image sample
- [ ] CLIP embedding generation + DBSCAN clustering for photo grouping
- [ ] Face detection/encoding for same-person grouping
- [ ] Date disambiguation logic (rule-based + cluster-based)
- [ ] Full pipeline run on all ~7,500 images
- [ ] Cloud API comparison pass (Gemini 2.5 Flash, GPT-4o-mini) — **needs user approval before calling**

## Failed Approaches (Don't Repeat These)

### Stamp detection v1: broad HSV color filter (stamp_detect.py as-is)

**What was attempted**: Scan edge regions of each photo for orange/red/amber pixels using HSV thresholds (H=0-40, S≥70, V≥120), find the largest contour, crop it.

**Why it failed**: 100/100 images "detected" a stamp — clearly false. Visual inspection of annotated outputs shows the detector is grabbing:
- Wood furniture/flooring (warm brown reads as orange in HSV)
- Skin tones near image edges
- Film scanning artifacts (orange strips along edges from the scanner)
- Colored toys, clothing, and objects (pink tubs, yellow toys)
- Dirt and mulch

**Example false positives verified by visual inspection:**
- `00000013` — detected wood/dirt in bottom-left; photo has NO date stamp at all
- `00000001` — detected film edge artifact on left side; actual stamp "10 4 '99" is on the right (photo is rotated 90°)
- `00000003` — detected some object, NOT the "10 3 '99" stamp in bottom-right

**One correct detection:** `00000005` — correctly found "9 28 '99" at bottom-center.

**Root cause**: The approach is too broad — "any warm pixel cluster near an edge" matches half the objects in a typical photo. Needs much stricter filtering:
- Require higher saturation (date stamps are vivid orange, not muted warm tones)
- Check that the cluster is SMALL relative to the image (date stamps are ~2-5% of image width)
- Check that detected region contains multiple discrete blobs (individual digits)
- Consider that date stamps have very consistent luminance (LED/LCD digits are uniformly bright)
- Should probably tighten to edge_pct=0.10 (bottom 10% not 15%)
- May want to look for digit-like contour shapes (roughly same height, spaced evenly)

**Why current approach is still the right foundation**: The CV-crop-then-OCR pipeline is sound — the crop step just needs much better filtering. An alternative is to skip CV entirely and send full images to an LLM, but that's 40x more expensive per image.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Use TrOCR instead of Tesseract as primary OCR | User requested it; transformer-based OCR handles scene text much better than Tesseract |
| CV crop + LLM hybrid approach (Approach D) | Best cost/quality tradeoff — $2-3 total vs $47+ for full-image LLM on all 7,500 |
| Venv at `.venv/` not system Python | System Python 3.14 uses PEP 668 (externally managed), can't pip install globally |
| Work from HDD source path, output to SSD | `originals/` and `needs_date/` are empty; source files are on HDD at `/mnt/823c.../Photos/img/Photos/ScanMyPhotos/` |
| No external API calls without user approval | User explicitly said "DO NOT call external APIs without my OK" |
| Local LLM via Ollama is OK | User approved Ollama for local inference |

## Current State

**Working**:
- Venv with all deps: `source .venv/bin/activate`
- `stamp_detect.py` runs and produces crops + annotated debug images
- Debug output at `debug_stamps/` with 100 images worth of crops, OCR-ready images, annotated originals, and `detection_results.json`

**Broken**:
- Stamp detection has near-100% false positive rate — needs rework before OCR makes sense
- TrOCR not yet wired up
- Ollama not installed

**Uncommitted Changes**: Not a git repo. All files are just on disk.

## Files to Know

| File | Why It Matters |
|------|----------------|
| `PLAN.md` | Master project plan for the entire 77K photo consolidation effort (Phases 1-3) |
| `DATE_EXTRACTION_APPROACHES.md` | Comprehensive analysis of 6 approaches with scores, costs, code snippets, full pipeline design |
| `stamp_detect.py` | Current stamp detection + cropping script — needs false-positive fixes |
| `debug_stamps/` | Output from first 100-image run — crops, annotated images, `detection_results.json` |
| `.venv/` | Python 3.14 venv with opencv, pillow, transformers, torch, torchvision |
| `metadata/` | Empty dirs for `albums/` and `ml_predictions/` |

## Code Context

**Current stamp detection signature** (`stamp_detect.py`):
```python
def detect_stamp_region(image_path: str, debug_dir: str | None = None) -> dict:
    """Returns: {found: bool, crop: np.ndarray|None, position: str, orange_pixel_count: int, ...}"""

def prepare_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """HSV filter + binarize + pad. Returns black-on-white image for OCR."""

def batch_detect_stamps(image_dir: str, debug_dir: str | None = None, limit: int | None = None) -> list[dict]:
    """Processes directory of JPGs, saves crops + OCR images to debug_dir."""
```

**Source photo paths**:
```
/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 1/  (1,775 files)
/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 2/  (2,040 files)
/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 3/  (2,076 files)
/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/Photos/img/Photos/ScanMyPhotos/Disc 4/  (1,576 files)
```

**Known date stamp examples** (from visual inspection):
```
Disc 2/00000003.jpg → "10 3 '99"  (bottom-right, orange)
Disc 2/00000023.jpg → "11 19 '99" (bottom-right, orange)
Disc 2/00000043.jpg → "12 3 '99"  (bottom-right, orange-red)
Disc 4/00000005.jpg → "4 18 '03"  (bottom-left, orange)
Disc 2/00000005.jpg → "9 28 '99"  (bottom-center, orange) — one correct CV detection
Disc 2/00000013.jpg → NO STAMP
Disc 1/00000003.jpg → NO STAMP
Disc 3/00000005.jpg → NO STAMP
```

**System specs** (affects local model choices):
```
CPU: AMD Ryzen (12 cores)
RAM: 27 GB
GPU: Integrated AMD Radeon Vega only (no discrete GPU)
Tesseract 5.3.4 installed system-wide
No Ollama installed yet
```

## Resume Instructions

1. **Activate venv**: `source /home/will/photo_project/.venv/bin/activate`

2. **Fix stamp detection** — the core issue. Strategies to try:
   - Tighten HSV: require S≥120 and V≥160 (stamps are vivid and bright, not muted)
   - Require the detected cluster to be small: width < 25% of image width, height < 10% of image height
   - Require aspect ratio 2.5:1 to 10:1 (date text is always much wider than tall)
   - Check for multiple sub-blobs within the cluster (individual digits, typically 5-8 of them)
   - Filter out large amorphous blobs (furniture/skin) — date stamp contours are compact and regular
   - Reduce edge_pct from 0.15 to 0.10
   - Validate by checking: photo 00000013 should return `found: False`, photo 00000005 should return `found: True` with crop of "9 28 '99"

3. **Re-run on 100 images** and visually verify crop quality on 10+ images

4. **Wire up TrOCR** on valid crops:
   ```python
   from transformers import TrOCRProcessor, VisionEncoderDecoderModel
   processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
   model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
   # Feed PIL images of crops → get text
   ```

5. **Install Ollama** for local LLM comparison:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen2.5-vl:7b
   ```

6. **Run comparison**: same 100 images through Tesseract, TrOCR, and Qwen2.5-VL — compare accuracy

7. **CLIP embeddings + clustering**: install `sentence-transformers`, generate embeddings, DBSCAN cluster

8. **Ask user before any external API calls** (Gemini, OpenAI, Anthropic)

## Warnings

- The HDD is mounted at `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/` — **never modify files there**. Read-only.
- `originals/`, `needs_date/`, `organized/` are all empty — the dedup pipeline (PLAN.md Phase 2) hasn't run yet. Work directly from the HDD source paths.
- Photo `00000001.jpg` in Disc 2 is rotated 90° clockwise — its date stamp appears on the right side. Many photos may be rotated. The stamp detector checks side edges too, but rotation detection would help.
- Some photos genuinely have no date stamp — the detector must be able to return `found: False`. The first run found 100/100 which proves this isn't working.
- `DATE_EXTRACTION_APPROACHES.md` has the full cost analysis, model comparisons, and recommended pipeline. Refer to it for architectural decisions.
- Python is 3.14 — some packages may have compatibility issues. The venv worked fine for current deps.
