import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM

# Device and dtype setup (added an mps fallback if you run this on a Mac)
if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
    print(f"Using CUDA GPU ({torch.cuda.get_device_name(0)}), dtype={torch_dtype}")
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
    print(f"Using Apple MPS, dtype={torch_dtype}")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print(f"Using CPU, dtype={torch_dtype}")

# Load model and processor (once)
model_id = "microsoft/Florence-2-large"
print(f"\nLoading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True, attn_implementation="eager").to(device)
print("  Model weights loaded.")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("  Processor loaded.")
print("Model ready.\n")

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
text_input = "small orange sequence of numbers in the corner"
prompt = task_prompt + text_input
print(f"Task:  {task_prompt}")
print(f"Query: {text_input}\n")

image_dir = Path("photo_mapping_samples")
images = sorted(p for p in image_dir.glob("*.jpg") if not p.stem.endswith("_boxes"))

print(f"Found {len(images)} images in {image_dir}/\n")
print("-" * 60)

for i, image_path in enumerate(images, 1):
    print(f"\n[{i}/{len(images)}] {image_path.name}")

    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.width}x{image.height}")

    print(f"  Running inference...", end="", flush=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=1,
        do_sample=False
    )
    print(" done.")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"  Raw output: {generated_text[:120]}{'...' if len(generated_text) > 120 else ''}")

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    results = parsed_answer.get(task_prompt, {})
    bboxes = results.get("bboxes", [])
    labels = results.get("labels", [])

    if bboxes:
        print(f"  Found {len(bboxes)} box(es):")
        for label, bbox in zip(labels, bboxes):
            print(f"    label={label!r}  bbox={[round(v) for v in bbox]}")
    else:
        print("  No boxes found.")

    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=3)

    output_path = image_path.with_stem(image_path.stem + "_boxes")
    image.save(output_path)
    print(f"  Saved: {output_path}")

print("\n" + "-" * 60)
print("Done.")
