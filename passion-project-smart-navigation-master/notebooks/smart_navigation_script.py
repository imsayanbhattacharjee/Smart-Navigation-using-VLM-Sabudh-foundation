

# ----------------------------------------!pip install transformers datasets torchvision torchaudio gtts accelerate  -q
!pip install datasets --upgrade

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
import os
from tqdm.auto import tqdm


from datasets import load_dataset
!pip install datasets --upgrade
ds = load_dataset("LearnItAnyway/Visual-Navigation-21k",split="train[:1000]")

ds

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def custom_data_collator(features):
    # First, let's process the image paths to actual images
    images = []
    valid_features = []  # To store features with successfully loaded images
    for feature in features:
        image_path = feature['image']
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            valid_features.append(feature)  # Add the feature if image loading is successful
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:  # Handle more potential errors
            print(f"Error loading image from path: {image_path}. Error: {e}")
            # Skip this feature and proceed to the next one

    # Check if valid_features is empty (no images loaded successfully)
    if not valid_features:
        print("All features in batch had image loading issues, skipping this batch")
        # Return an empty dictionary with the necessary keys, but empty tensors
        return {
            'input_ids': torch.empty((0, 0), dtype=torch.long),
            'attention_mask': torch.empty((0, 0), dtype=torch.long),
            'labels': torch.empty((0, 0), dtype=torch.long),
            'pixel_values': torch.empty((0, 3, 224, 224), dtype=torch.float32), # Assuming standard BLIP input size
        }

    # Now, process the images and instructions using the BLIP processor
    # Access the 'description' key, as it is the correct label key in this dataset
    inputs = processor(images=images, text=[feature["description"] for feature in valid_features], return_tensors="pt", padding=True)

    # Remove the original 'image' key and add pixel values
    del inputs['image']
    inputs['pixel_values'] = inputs.pop("images")  # Rename 'images' key to 'pixel_values'

    # Add labels to the inputs dictionary
    inputs['labels'] = inputs['input_ids'].detach().clone()

    return inputs

training_args = TrainingArguments(
    output_dir="./blip-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Set True to use Colab GPU efficiently
    report_to="none"  # Disable wandb integration
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=custom_data_collator
)

# Start the training process
trainer.train()

# After the trainer.train() call, save the processor:
trainer.save_model("./blip-finetuned") # This will save the model
processor.save_pretrained("./blip-finetuned") # This will save the processor

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load fine-tuned processor and model
processor = BlipProcessor.from_pretrained("./blip-finetuned")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 🔁 Replace with your uploaded filename if different
image_path = "team.jpg"  # 👈 change this if your image has another name
image = Image.open(image_path).convert("RGB")

# Generate caption
inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

# Show result
print("📝 Generated Caption:", caption)
image.show()


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import os

# Load fine-tuned model
processor = BlipProcessor.from_pretrained("./blip-finetuned")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Set video path
video_path = "cars.mp4"  # ✅ replace this if your filename is different
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
print(cap)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps  # Capture 1 frame per second
frame_idx = 0
captions = []

print("⏳ Generating captions for video frames...\n")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        # Convert to RGB and PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Run BLIP captioning
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # Save image + caption
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx}.jpg")
        pil_image.save(frame_path)
        captions.append((frame_path, caption))
        print(f"🖼️ Frame {frame_idx // frame_interval}: {caption}")

    frame_idx += 1

cap.release()

# Summary
print("\n✅ All captions generated:")
for path, cap_text in captions:
    print(f"{os.path.basename(path)}: {cap_text}")


from IPython.display import display
from google.colab import output
from PIL import Image
import io
import IPython
import cv2
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model and processor
processor = BlipProcessor.from_pretrained("./blip-finetuned")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# JavaScript to capture an image from webcam
def capture_image():
    js = """
    async function capture() {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = '📸 Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      video.width = 320;
      video.height = 240;
      div.appendChild(video);

      document.body.appendChild(div);

      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();

      // Resize video to fit container
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getTracks().forEach(track => track.stop());
      div.remove();

      const dataUrl = canvas.toDataURL('image/jpeg');
      return dataUrl;
    }
    capture();
    """
    return output.eval_js(js)

def data_url_to_image(data_url):
    header, encoded = data_url.split(",", 1)
    binary = io.BytesIO(base64.b64decode(encoded))
    img = Image.open(binary)
    return img

import base64

# Capture image
data_url = capture_image()
image = data_url_to_image(data_url)

# Show captured image
display(image)

# Preprocess and caption
inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(output[0], skip_special_tokens=True)

print("Generated Caption:", caption)


from gtts import gTTS
from IPython.display import Audio

#Use the `caption` variable from the generated BLIP output
caption_text = caption  # your generated caption
print("Generated Caption:", caption_text)

# 🎧 Convert text to speech
tts = gTTS(text=caption_text, lang='en')
tts.save("caption_audio.mp3")

# Play audio in Colab
Audio("caption_audio.mp3")


!pip install gTTS opencv-python-headless transformers -q
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
from IPython.display import Audio, display
import torch
import cv2
import os

# Load BLIP model
processor = BlipProcessor.from_pretrained("./blip-finetuned")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Video path
video_path = "cars.mp4"  #  Replace with your uploaded video
output_audio_dir = "audio_outputs"
os.makedirs(output_audio_dir, exist_ok=True)

# Extract frames using OpenCV
cap = cv2.VideoCapture(video_path)
frame_rate = 1  # 1 frame per second
frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) * frame_rate
frame_count = 0
caption_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Convert frame to PIL image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Caption generation
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        print(f" Frame {frame_count} Caption:", caption)

        # Convert to audio
        tts = gTTS(text=caption, lang='en')
        audio_file = os.path.join(output_audio_dir, f"caption_{caption_count}.mp3")
        tts.save(audio_file)

        # Optional: Play audio in notebook
        display(Audio(audio_file))

        caption_count += 1

    frame_count += 1

cap.release()


!pip install gradio gtts transformers torchvision opencv-python


import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from gtts import gTTS
import tempfile
import os

# Load fine-tuned model
processor = BlipProcessor.from_pretrained("./blip-finetuned")
model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def caption_and_audio(image):
    # Generate caption
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Generate audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tts = gTTS(text=caption)
        tts.save(tmp_audio.name)
        audio_path = tmp_audio.name

    return caption, audio_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Image Captioning + Audio for Visually Impaired (BLIP + gTTS)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        submit_btn = gr.Button("Generate Caption & Audio")

    caption_output = gr.Textbox(label="📝 Generated Caption", interactive=False)
    audio_output = gr.Audio(label="🔊 Audio Output")

    submit_btn.click(fn=caption_and_audio, inputs=image_input, outputs=[caption_output, audio_output])

demo.launch(share=True)


import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
from gtts import gTTS
import os
import tempfile
import zipfile


def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def caption_image(img):
    caption = generate_caption(img)
    tts = gTTS(text=caption, lang='en')
    tts_path = tempfile.mktemp(suffix=".mp3")
    tts.save(tts_path)
    return caption, tts_path

def caption_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    captions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = generate_caption(img)
            captions.append(caption)
        frame_count += 1
    cap.release()
    full_caption = " ".join(captions[:5])
    tts = gTTS(text=full_caption, lang='en')
    tts_path = tempfile.mktemp(suffix=".mp3")
    tts.save(tts_path)
    return full_caption, tts_path

def caption_webcam(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    caption = generate_caption(img)
    tts = gTTS(text=caption, lang='en')
    tts_path = tempfile.mktemp(suffix=".mp3")
    tts.save(tts_path)
    return caption, tts_path


import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("## Smart Visual Captioning System for the Visually Impaired")

    with gr.Tab("📸 Image"):
        img_input = gr.Image(type="pil")
        img_caption = gr.Textbox(label="Generated Caption")
        img_audio = gr.Audio(label="Audio Output")
        img_button = gr.Button("Generate Caption")
        img_button.click(fn=caption_image, inputs=img_input, outputs=[img_caption, img_audio])

    with gr.Tab("Video"):
        vid_input = gr.Video()
        vid_caption = gr.Textbox(label="Video Summary Caption")
        vid_audio = gr.Audio(label="Audio Output")
        vid_button = gr.Button("Generate Caption")
        vid_button.click(fn=caption_video, inputs=vid_input, outputs=[vid_caption, vid_audio])

    with gr.Tab("📷 Webcam (Simulated in Colab)"):
        cam_input = gr.Image(label="Take or Upload a Photo", type="pil")
        cam_caption = gr.Textbox(label="Live Caption")
        cam_audio = gr.Audio(label="Audio Output")
        cam_button = gr.Button("Generate Caption")
        cam_button.click(fn=caption_image, inputs=cam_input, outputs=[cam_caption])


    demo.launch(debug=True)


