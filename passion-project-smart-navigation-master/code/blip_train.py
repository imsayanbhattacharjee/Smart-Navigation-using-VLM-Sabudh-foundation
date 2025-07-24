import os
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# === CONFIGURATION ===
IMAGE_FOLDER = "images"     #your folder name
CAPTION_FOLDER = "captions" #your folder name
CAPTIONS_PER_IMAGE = 5 
NUM_EPOCHS = 5
BATCH_SIZE = 4

# === LOAD IMAGE AND CAPTION FILES ===
image_files = sorted(os.listdir(IMAGE_FOLDER))
caption_files = sorted(os.listdir(CAPTION_FOLDER))  

# Make sure both folders have the same number of files
if len(image_files) != len(caption_files):
    raise ValueError(f"Number of images ({len(image_files)}) and captions ({len(caption_files)}) do not match.")

# === BUILD DATASET ===
data = []

for i in range(len(image_files)):
    # Image file
    image_name = image_files[i]
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    
    # Corresponding caption file
    caption_name = caption_files[i]
    caption_path = os.path.join(CAPTION_FOLDER, caption_name)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        continue
    if not os.path.exists(caption_path):
        print(f"Warning: Caption not found: {caption_path}")
        continue

    # Read all captions from the text file
    with open(caption_path, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f.readlines() if line.strip()]
    
    # If we have more captions than the specified number, we'll trim it
    if len(captions) > CAPTIONS_PER_IMAGE:
        captions = captions[:CAPTIONS_PER_IMAGE]
    
    for caption in captions:
        data.append({"image_path": image_path, "caption": caption})

dataset = Dataset.from_list(data)
print(f"Total training examples: {len(dataset)}")

# === LOAD MODEL AND PROCESSOR ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# === PREPROCESS FUNCTION ===
def preprocess(batch):
    images = [Image.open(path).convert("RGB").resize((384, 384)) for path in batch["image_path"]]
    inputs = processor(images=images, text=batch["caption"], padding="max_length",
                       return_tensors="pt", max_length=128, truncation=True)

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# === MAP WITH MEMORY-EFFICIENT SETTINGS ===
dataset = dataset.map(
    preprocess,
    batched=True,
    batch_size=4,  # Lower = safer
    keep_in_memory=False,
    load_from_cache_file=False
)

# === TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir="./blip-finetuned",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=100,
    save_total_limit=2,
)

# === TRAINING LOOP ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer
)

trainer.train()

print("✅ Training complete! Model saved to ./blip-finetuned")
