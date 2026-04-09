# YOLO date stamp detector

default:
    @just --list

# Train the model (resumes from previous best.pt if available)
train:
    uv run train.py

# Run batch inference on pending images
infer:
    uv run infer_all.py

# Train then infer (full cycle)
cycle: train infer

# Start annotation server on :8888
annotate:
    uv run annotate.py

# Start annotation in correction mode on :8888
annotate-correct:
    uv run annotate.py --mode correct

# Start the corrections dashboard on :8889
dashboard:
    uv run corrections_dashboard.py

# Run OCR on detected stamps (requires ANTHROPIC_API_KEY)
ocr *ARGS:
    uv run ocr_stamps.py {{ARGS}}

# One-time setup: copy ScanMyPhotos images to working directory
setup-scanmyphotos:
    uv run setup_scanmyphotos.py

# Run inference on a single photo
infer-one photo conf="0.35":
    uv run python -c "\
    from ultralytics import YOLO; \
    r = YOLO('runs/detect/train/weights/best.pt')('{{photo}}', imgsz=384, conf={{conf}}, device='cpu', verbose=False)[0]; \
    [print(f'conf={float(b.conf[0]):.4f} bbox={[round(float(x),3) for x in b.xyxy[0]]}') for b in r.boxes]; \
    print('No detections') if len(r.boxes) == 0 else None; \
    r.save('infer_output.jpg'); \
    print('Saved: infer_output.jpg'); \
    "

# Show dataset statistics
stats:
    #!/usr/bin/env python3
    from pathlib import Path
    import json
    labels = {p.stem for p in Path('dataset/labels').glob('*.txt') if p.stem.startswith('d')}
    skipped = set()
    if Path('skipped.txt').exists():
        skipped = {l.strip() for l in open('skipped.txt').readlines() if l.strip()}
    image_stems = {p.stem for p in Path('scanmyphotos').glob('*.jpg')} if Path('scanmyphotos').exists() else set()
    preds = len(json.load(open('scanmyphotos_predictions.json'))) if Path('scanmyphotos_predictions.json').exists() else 0
    reviewed = (labels | skipped) & image_stems
    has_stamp = labels & image_stems
    no_stamp = skipped & image_stems
    remaining = len(image_stems) - len(reviewed)
    pct = (len(reviewed) / len(image_stems) * 100) if image_stems else 0
    print(f'Images:      {len(image_stems)}')
    print(f'Has stamp:   {len(has_stamp)}')
    print(f'No stamp:    {len(no_stamp)}')
    print(f'Reviewed:    {len(reviewed)} / {len(image_stems)} ({pct:.1f}%)')
    print(f'Remaining:   {remaining}')
    print(f'Predictions: {preds}')

# Update status.json with current stats
update-status:
    #!/usr/bin/env python3
    import json
    from pathlib import Path
    from datetime import date
    labels = list(Path('dataset/labels').glob('*.txt'))
    labels_d = [l for l in labels if l.stem.startswith('d')]
    skipped = set()
    if Path('skipped.txt').exists():
        skipped = {l.strip() for l in open('skipped.txt').readlines() if l.strip()}
    images = list(Path('scanmyphotos').glob('*.jpg')) if Path('scanmyphotos').exists() else []
    preds = json.load(open('scanmyphotos_predictions.json')) if Path('scanmyphotos_predictions.json').exists() else {}
    queue = json.load(open('corrections_queue.json')) if Path('corrections_queue.json').exists() else {'stats': {}}
    qs = queue.get('stats', {})
    manifest = json.load(open('scanmyphotos_manifest.json')) if Path('scanmyphotos_manifest.json').exists() else []
    disc_counts = {}
    for m in manifest:
        d = m['disc']
        disc_counts[d] = disc_counts.get(d, 0) + 1
    total = len(images) or len(manifest)
    reviewed = qs.get('labeled', 0) + qs.get('no_stamp', 0)
    train_imgs = len(list(Path('dataset/images/train').glob('*.jpg'))) if Path('dataset/images/train').exists() else 0
    val_imgs = len(list(Path('dataset/images/val').glob('*.jpg'))) if Path('dataset/images/val').exists() else 0
    status = {
        'generated_at': str(date.today()),
        'source_images': {'total': total, **{f'disc_{d}': c for d, c in sorted(disc_counts.items())}},
        'annotation': {
            'labeled_with_stamp': qs.get('labeled', len(labels_d)),
            'confirmed_no_stamp': qs.get('no_stamp', len(skipped)),
            'skipped': qs.get('skipped', 0),
            'pending_review': qs.get('pending', 0),
            'percent_reviewed': round(reviewed / total * 100, 1) if total else 0,
        },
        'training': {
            'labels_total': len(labels),
            'labels_disc_prefixed': len(labels_d),
            'train_images': train_imgs,
            'val_images': val_imgs,
            'base_model': 'yolov8n.pt',
            'training_config': {'epochs': 100, 'patience': 10, 'imgsz': 640, 'batch': 8, 'device': 'cpu'},
        },
        'inference': {
            'predictions_total': len(preds),
            'inference_size': 384,
            'confidence_threshold': 0.01,
        },
    }
    with open('status.json', 'w') as f:
        json.dump(status, f, indent=2)
    print(f'Updated status.json ({reviewed}/{total} reviewed, {len(preds)} predictions)')

# Show training metrics in TensorBoard
tensorboard:
    uvx --python 3.12 --with "setuptools<82" tensorboard --logdir runs/detect

# Feedback loop: prepare correction images
feedback-prepare:
    uv run feedback.py prepare

# Feedback loop: finalize corrections
feedback-finalize:
    uv run feedback.py finalize

# Feedback loop: show status
feedback-status:
    uv run feedback.py status
