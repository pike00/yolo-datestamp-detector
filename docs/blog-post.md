# Teaching a Neural Net to Find Date Stamps on Scanned Photos

Or: how I went from "this should be a regex" to "fine-tune a YOLO model" to "why is this taking 58 days to train."

## The problem

I have roughly 77,000 family photos spread across an old HDD. About 7,500 of them are scans from a service called ScanMyPhotos -- four DVDs, shoeboxes of 4x6 prints from the 90s and early 2000s. The scans are fine. The problem is the metadata: every JPG file has a `DateCreated` of whenever the scanner operator fed it into the machine, which is useless. A photo of my kindergarten graduation shows up in the library as "2018".

The good news: a lot of these photos have date stamps burned right onto the film. If you remember disposable cameras and early point-and-shoots, you remember the little orange LED digits in the corner:

```
10 3 '99
```

That is the real date. If I can read those stamps, I can write them back into EXIF and the whole library snaps into chronological order. So the question reduces to: given a scan, find the date stamp, then read it.

This post is about the "find it" half. The reading (OCR) is a different story.

## Attempt 1: OpenCV heuristics

The first instinct was "this is a computer vision toy problem, not a machine learning problem." The stamps are always orange-ish, always small, always near an edge. Surely I can just threshold on HSV and find them.

Here is roughly what I tried:

1. Convert to HSV.
2. Mask pixels in the orange/amber range (hue 10-25, saturation > 100, value > 100).
3. Dilate, find contours, filter by aspect ratio (wider than tall, reasonable size).
4. Assume the largest orange blob near the bottom edge is the stamp.

It worked beautifully on the first 20 photos I tested. It fell apart on the 21st, which was a picture of a sunset over the Grand Canyon. The entire bottom half of the image was orange. The "date stamp" the detector found was a cliff.

The failure modes kept accumulating:

- Orange flowers, orange shirts, orange sunsets, orange Christmas lights.
- Stamps on bright backgrounds (overexposed sky) that washed out and didn't cross the saturation threshold.
- Photos rotated 90 degrees where the stamp was on the side, not the bottom.
- Photos with *no* stamp at all, which the heuristic would confidently produce a garbage box for anyway.
- Faded scans where the stamp was more brown than orange.

I could have kept bolting on exceptions. Edge case handling in CV pipelines tends to metastasize, and I could already see the shape of it: per-photo tuning, manual overrides, an ever-growing list of `if sunset: skip`. Not fun.

## Attempt 2: Fine-tune YOLO

Object detection is the textbook solution for "find a bounding box around a known thing in a photo." YOLO (You Only Look Once) is a family of single-pass detectors from Ultralytics that are fast, well-tooled, and come with pretrained weights on COCO. Fine-tuning one of these on a few thousand labeled stamps should crush the heuristic approach, and I wouldn't have to think about sunsets at all.

I picked YOLOv8-nano to start, because (a) I'm on a desktop with no discrete GPU (AMD Ryzen 5 5600G, integrated Vega graphics that torch can't talk to) and (b) the task is stupid simple from a detector's perspective: one class, distinctive color, consistent shape. The nano variant is 3 million parameters, which is basically nothing by 2026 standards. It should be overkill.

## Building an annotation UI

The first real problem: there are no labels. YOLO wants a `.txt` file per image with normalized `class cx cy w h` lines. I need to draw bounding boxes on a few thousand scans.

I built a tiny browser-based annotator ([scripts/annotate.py](../scripts/annotate.py) + [ui/index.html](../ui/index.html)). It's a Python HTTP server exposing a REST API and serving a vanilla-JS Canvas frontend. Features I ended up caring about, in order of how much they mattered:

1. **Keyboard-first workflow.** Arrow keys to move between images, `s` to skip, `enter` to save, `z` to undo. Mouse only for drawing the box. Labeling thousands of photos gets painful fast if you have to click buttons.
2. **Skip as signal.** When I marked a photo "no stamp here," the stem got written to `state/skipped.txt`. These become *negative* training examples -- photos with no label file but present in the dataset. YOLO handles this correctly (it treats them as "nothing to detect here") and they matter a lot for precision.
3. **Auto-advance.** When I saved a box, the next image loaded immediately. No confirmation dialogs, no "are you sure." I can afford to mislabel one out of a thousand; I cannot afford a two-second stall per photo.
4. **Persistent state.** The server tracked my position, so when I got bored and closed the tab, the next session picked up where I left off.

Edge case that took an hour to diagnose: my first annotator session produced labels named `00000080.txt`, matching the original scan filenames. I later reorganized source photos with disc prefixes (`d1_00000080.jpg`) so I could tell Disc 1 from Disc 3, and suddenly every label was orphaned. I wrote a migration block in [scripts/train.py:33-64](../scripts/train.py#L33-L64) that walks old-style label filenames and rewrites them to match disc-prefixed image stems. Keeping it in `setup_dataset()` is ugly but it runs every train and means I never have to think about it again.

## Labels and the 80/20 split

After a few evenings of clicking, I had around 2,600 labeled images and another ~300 "skipped" negatives. Split 80/20 into train and val with `random.seed(42)` so splits are reproducible across training runs. Symlink the images into `dataset/images/{train,val}/` and copy the labels into `dataset/labels/{train,val}/`.

Note on the copy-vs-symlink choice: I symlink images (they're big, I don't want two copies) and *copy* labels (they're tiny, and symlinking them caused weird permissions issues when the dataset got mounted into a Docker container). Small detail; mattered a lot when training refused to start with `PermissionError: [Errno 13]` on day one.

## The first real training run

With ~2,900 examples, I kicked off a training run:

```python
model.train(
    data="dataset/data.yaml",
    epochs=100,
    patience=10,
    imgsz=640,
    batch=4,
    device="cpu",
    ...
)
```

Epoch 1 took about 14 minutes. Epoch 2 was faster. By epoch 15 I was at mAP50 = 0.89. By epoch 27 I had:

| Metric        | Value |
| ------------- | ----- |
| Precision     | 95.3% |
| Recall        | 95.8% |
| mAP@50        | 95.0% |
| mAP@50-95     | 73.8% |
| F1 (peak)     | 0.96 at conf 0.37 |

Training early-stopped at epoch 37. That 73.8% mAP@50-95 was the only result I wasn't thrilled about -- it means the model finds the stamps but its bounding boxes are a little loose. For my purposes that is fine: the downstream OCR only needs a rough crop that contains the stamp, and I'd rather have a loose box that includes the whole stamp than a tight one that clips half of a digit.

## Batch inference and the first reality check

Running the trained model across ~7,500 scans took about 40 minutes on CPU with `imgsz=384` (smaller than training, deliberately -- inference doesn't need full resolution). I got 6,458 detections at `conf >= 0.01`.

I built a corrections dashboard ([scripts/corrections_dashboard.py](../scripts/corrections_dashboard.py), [ui/dashboard.html](../ui/dashboard.html)) to review the results. Same keyboard-first vibe as the annotator, but now the model proposes a box and I either confirm it, nudge it, or mark it as wrong.

The confidence histogram was bimodal: a big peak near 0.8 (obvious correct detections) and a smaller cluster around 0.3-0.4 (borderline cases worth reviewing). Anything above 0.7 I could bulk-approve with a glance. Anything below 0.5 I actually looked at.

The failure modes I saw:

- **Bright skies washing out the stamp.** The model just missed these. The stamp was there, the colors were right, but the contrast was gone.
- **Rotated photos.** A scan flipped 90 degrees has the stamp on the side, and the model -- which saw 99% of its training data in a consistent orientation -- would either miss it or confidently predict a region of empty sky.
- **Orange content near an edge.** Orange curtains, a child's orange sweater right at the bottom of the frame. The model got better than the HSV heuristic at ignoring these, but not perfect.
- **No-stamp photos with spurious detections.** This is what negative examples are for, and it was *much* worse before I added the skipped-photo pipeline. Zero false positives on background images after I added them -- the confusion matrix came back completely clean.

## Hard-case augmentation: the bright-background fix

I built [scripts/augment_hard_cases.py](../scripts/augment_hard_cases.py) specifically to fight the bright-background failure mode. The idea: take the photos I already labeled and synthesize augmented copies with the characteristics the model was missing. Brightness up 1.6x and 2.0x. Contrast down to 0.5x. Gamma corrections. Warm and cool color-temperature shifts. Since all the transforms are global (no rotation, no crop), the bounding box labels are valid on the augmented copies without any remapping.

Quick implementation notes that paid off:

- Pre-compute a gamma LUT. Per-pixel `float ** 0.6` was the hot path; a 256-entry uint8 lookup table is a hundred times faster.
- Do everything in numpy, not PIL. PIL is fine but converting back and forth per augmentation type was dominating the loop.
- Use `ProcessPoolExecutor` across the 12 cores. Augmentation is embarrassingly parallel and single-threaded numpy leaves 11 cores bored.
- Write labels as a hardlink or copy, not a regeneration. They're identical for every augmentation.

Run the full augmentation over ~2,600 labeled images producing 6 variants each, and you get about 15,000 augmented images. Add them to the training split (never to val -- validation must remain independent). The augmented data went under `dataset/augmented/` with a separate labels dir so I could nuke them with `just augment-clean` whenever I wanted.

And this is where things started to get slow.

## The 58-day training run

I tried to fine-tune again after adding the augmented set. `just train` printed a cheerful "Training with 17,824 images." First epoch kicked off. I checked the progress bar:

```
Epoch 1/100:   3%|▎         | 134/4456 [25:21<14:03:22, 11.4s/it]
```

Fourteen hours per epoch. Times 100 epochs. Times... wait.

```
14 * 100 / 24 = 58 days
```

Reader, I did not have 58 days. I had maybe an overnight. Things I considered, briefly:

- Just let it run. (No. Power bill alone. Plus, overfitting risk if I just let it thrash.)
- Kill it and accept the previous model. (The previous model had exactly the failure modes the augmented set was supposed to fix.)
- Rent a GPU. (Tempting. Still tempting.)
- Figure out why it was so slow and shave the cost down.

I went with the last one, and in retrospect the first thing I should have done is *not* start a run I hadn't sanity-checked. Eyeballing the numbers before hitting go would have caught this in 30 seconds.

## Root causing the slowdown

Five things were conspiring:

**1. Augmentation inflated the dataset 7x.**
Before: 2,651 real training images. After: 2,651 + 15,175 augmented = 17,824. Every epoch now paid a 7x multiplier, and the augmentation wasn't even the bottleneck I was trying to fix -- I'd written it to fix a precision issue, not as a dataset-size strategy. The vast majority of those augmented images were routine variations the model would learn from the real data anyway.

Fix: `just augment-clean`, and don't re-add augmentations until the basic run is cheap again. When I do re-add them, subsample to maybe 2-3x not 7x.

**2. Model was `yolo26s`, not `yolo26n`.**
The nano variant is ~3M parameters; small is ~9.9M. I'd bumped it up at some point and forgotten. For a single-class detector where the target is a high-contrast rectangular region, nano is almost certainly sufficient. Small-vs-nano accounted for maybe a 3x slowdown.

Fix: switch back to `yolo26n.pt`. Make the choice a CLI flag so I can't silently regress again.

**3. `imgsz=640`.**
YOLO CPU cost is roughly quadratic in image size. 640 vs 416 is (640/416)^2 ≈ 2.4x. My actual stamps are tiny relative to the photo, but they're also distinctive enough that the model finds them at inference with `imgsz=384`. Training at 640 was flattering the metrics without earning its cost.

Fix: drop to 416 for training.

**4. `epochs=100` with `patience=10`.**
This is fine in principle -- patience will stop training if val loss plateaus. But my *very first epoch* came back with mAP50 = 0.951. The model was converging in 1-2 epochs. Asking for 100 and letting patience stop me was needlessly generous.

Fix: cap at 40 epochs. Patience will still kick in.

**5. `workers=0`.**
I noticed in the Ultralytics source that they explicitly force `workers=0` on CPU:

```python
# ultralytics/engine/trainer.py
if self.device.type in {"cpu", "mps"}:
    self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading
```

I went down a rabbit hole trying to override this before realizing: on a CPU-bound run, the GIL and the cache locality story mean more dataloader workers actively hurt. Ultralytics knows what it's doing. This was the hours-spent-on-the-wrong-thing of the afternoon, and the comment in their source code is the reason I stopped.

Not a fix. A non-issue. Move on.

## Stacked speedup estimate

```
 7x  (drop augmentation)
x3x  (nano vs small)
x2.4x (imgsz 416 vs 640)
x3x  (40 epochs vs 100, since we'll early-stop anyway)
~150x total
```

58 days / 150 ≈ 9 hours. Overnight. Done.

Alternative: rent a T4 or RTX 4000 at $0.50/hour, rsync the dataset up, and the original config finishes in 30 minutes for about a quarter. The reason I'm writing this post is partly that I keep *not* doing this and it keeps being a question I could just settle with a credit card.

## CLI flags for the things that should always have been CLI flags

The underlying bug was that model, epochs, imgsz, and batch were all hardcoded in [scripts/train.py](../scripts/train.py). Iterating on them meant editing the script, which meant git diffs, which meant I would stop iterating and pick a plausible-sounding default and move on. I added argparse flags for all of them, wired them through the `justfile` recipe (which already used `*ARGS`), and now `just train --model yolo26n.pt --no-aug --imgsz 416 --epochs 40` does what it says.

Moral: if you find yourself considering a default, you probably want a flag.

## One last landmine: the resume logic

[train.py:223-229](../scripts/train.py#L223-L229) has this:

```python
best_pt = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
if best_pt.exists():
    model = YOLO(str(best_pt))
else:
    model = YOLO("yolo26s.pt")
```

This is great for iterative fine-tuning from previous best weights, and it is a *trap* when you switch model architectures. Your brand new `--model yolo26n.pt` flag gets silently ignored because `best.pt` exists and takes precedence. I learned this the slow way: started a "nano" run, watched the first batches and thought "huh, that's not any faster," and realized it had resumed from the old small model.

Fix: move or delete `runs/detect/train/weights/best.pt` before any run where you change model size. Or better, make the resume step respect the `--model` flag and error out if the architectures don't match. I did the fix the lazy way (move the file); the better fix is a TODO.

## What it looks like working

Pipeline end-to-end now:

1. `just annotate` -- label 50 more photos in 20 minutes.
2. `just train --no-aug --model yolo26n.pt --epochs 40` -- train overnight on CPU.
3. `just infer` -- batch inference across all 7,500 scans in about 40 minutes.
4. `just dashboard` -- review predictions, correct the borderline cases, confirm the easy ones. A few hours of focused clicking.
5. Feed corrections back into `dataset/corrections/`, re-train.
6. `just ocr` -- crop each detected region and hand it to Claude Haiku (or a local Gemma model via Ollama) for text extraction. Output is a JSON mapping stem to parsed date.
7. Write the date back into EXIF with a separate script in the parent photo project.

The detection part is basically solved. The remaining uncertainty lives in the OCR step, where "4 23 '95" and "4 23 '85" look very similar on a faded scan, and is a different blog post entirely.

## Things I'd do differently

- **Sanity-check training runtime with a 1-epoch dry run before committing to the full config.** 30 seconds of arithmetic would have saved me from the 58-day fiasco.
- **Make every training hyperparameter a CLI flag from day one.** Hardcoded defaults become hidden state.
- **Build the negative-example pipeline before training, not after.** The first model had enough false positives that I almost gave up on YOLO. Adding "skipped" photos as background examples fixed it in one training run.
- **Augment surgically, not generously.** The 7x augmentation explosion was me pattern-matching "more data is better" when what I actually needed was targeted bright-background samples. A 2x augmentation focused on the actual failure mode would have been more effective *and* cheaper to train.
- **Just rent the GPU.** Seriously. $0.50.

## Things I'd keep

- **Keyboard-first annotation UI.** The single best return on investment in the whole project. Labeling is the bottleneck; ergonomics of labeling is the bottleneck of the bottleneck.
- **Corrections dashboard that feeds back into training.** Active learning in its simplest form. The model gets better every week I use it, not because I retrained it on a bigger dataset but because I retrained it on the images where it was *wrong*.
- **Ultralytics' defaults.** Every time I went to override one, I either ended up putting it back or finding a comment in their source explaining why they were right. The project's opinions are load-bearing and mostly correct.
- **The apprise notifications.** Getting a Mattermost ping every 10 epochs meant I could kick off a run, walk away, and find out if it crashed without babysitting TensorBoard. Tiny feature, disproportionate quality-of-life.

## Final numbers

| Thing | Value |
| ----- | ----- |
| Photos to process | ~7,500 |
| Labeled by hand | ~2,600 |
| Negative examples | ~300 |
| Base model | YOLOv8-nano (~3M params) |
| Training time (CPU, no aug, 40 epochs) | ~9 hours |
| Inference time (all photos) | ~40 minutes |
| Precision / Recall / mAP50 | 95.3% / 95.8% / 95.0% |
| Cost to run on cloud GPU instead | roughly $0.25 |
| Sunsets incorrectly identified as dates | 0 |

The sunset count is the metric I'm proudest of.
