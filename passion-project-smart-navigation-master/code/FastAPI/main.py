# main.py
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageEnhance, ImageFilter
import io
import cv2
import os
import uuid
import numpy as np
from typing import List
import logging

# Add this after the imports
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add this after device setup
processing_status = {}
executor = ThreadPoolExecutor(max_workers=2)

# Initialize FastAPI app
app = FastAPI()

# Setup paths
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enhanced model loading with better device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load fine-tuned BLIP model with optimized settings
try:
    processor = BlipProcessor.from_pretrained("./blip-finetuned2")
    
    # Load model with appropriate dtype based on device
    if device == "cuda":
        model = BlipForConditionalGeneration.from_pretrained(
            "./blip-finetuned2", 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = BlipForConditionalGeneration.from_pretrained(
            "./blip-finetuned2",
            torch_dtype=torch.float32
        ).to(device)
    
    model.eval()
    logger.info("Fine-tuned BLIP model loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load fine-tuned model: {e}")
    # Fallback to base model if fine-tuned model fails
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    logger.info("Fallback to base BLIP model")

speak_enabled = True

def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhanced image preprocessing for better accuracy."""
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to optimal size (keeping aspect ratio)
        max_size = 512
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Enhance image quality
        # Slightly enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Slightly enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Apply slight noise reduction
        image = image.filter(ImageFilter.MedianFilter(size=1))
        
        return image
    except Exception as e:
        logger.warning(f"Image preprocessing failed, using original: {e}")
        return image.convert('RGB') if image.mode != 'RGB' else image

def generate_caption_with_beam_search(image: Image.Image, prompt: str = None, is_video_frame: bool = False) -> str:
    """Generate caption using beam search for better accuracy."""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Prepare inputs
        if prompt:
            inputs = processor(processed_image, prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(images=processed_image, return_tensors="pt").to(device)
        
        # Adjust parameters for video frames (faster processing)
        if is_video_frame:
            max_tokens = 30
            num_beams = 3
            temp = 0.8
        else:
            max_tokens = 50
            num_beams = 5
            temp = 0.7
        
        # Generate with improved parameters - FIXED: removed duplicate pad_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=True,
                temperature=temp,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                length_penalty=1.0,
                no_repeat_ngram_size=2
                # Removed pad_token_id since it might already be in inputs
            )
        
        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        # Remove prompt from caption if it was used
        if prompt and caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip()
        
        # Post-process caption
        caption = post_process_caption(caption)
        
        return caption
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        return "Unable to generate caption."

def post_process_caption(caption: str) -> str:
    """Post-process generated caption for better quality."""
    if not caption:
        return "No caption generated."
    
    # Remove common artifacts
    caption = caption.replace("arafed", "").replace("araffe", "")
    
    # Capitalize first letter
    caption = caption.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    
    # Ensure it ends with proper punctuation
    if caption and caption[-1] not in '.!?':
        caption += '.'
    
    return caption

def extract_key_frames(video_path: str, max_frames: int = 15) -> List[Image.Image]:
    """Extract key frames from video using improved algorithm."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count == 0:
        cap.release()
        return frames
    
    # Calculate frame extraction strategy
    if frame_count <= max_frames:
        # Extract every frame if video is short
        frame_indices = list(range(frame_count))
    else:
        # Extract frames at regular intervals
        interval = frame_count // max_frames
        frame_indices = [i * interval for i in range(max_frames)]
    
    prev_frame = None
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Skip very similar frames (basic motion detection)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            diff_score = np.mean(diff)
            if diff_score < 10:  # Skip if frames are too similar
                continue
        
        frames.append(Image.fromarray(frame_rgb))
        prev_frame = frame
    
    cap.release()
    return frames

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/caption-image/")
async def caption_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate detailed caption
        caption = generate_caption_with_beam_search(image)
        
        logger.info(f"Generated image caption: {caption}")
        return {"caption": caption}
        
    except Exception as e:
        logger.error(f"Image captioning failed: {e}")
        return {"caption": "Image processing failed."}

def process_video_frames(temp_path: str, task_id: str):
    """Process video frames in a separate thread with status updates."""
    try:
        processing_status[task_id] = {"status": "extracting_frames", "progress": 10}
        
        # Extract key frames
        frames = extract_key_frames(temp_path, max_frames=8)
        
        if not frames:
            processing_status[task_id] = {"status": "error", "message": "Unable to extract frames"}
            return {"caption": "Unable to extract frames from video."}
        
        processing_status[task_id] = {"status": "processing_frames", "progress": 20, "total_frames": len(frames)}
        
        # Generate captions for key frames
        frame_captions = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            try:
                progress = 20 + (i / total_frames) * 60  # 20% to 80% progress
                processing_status[task_id] = {
                    "status": "processing_frames", 
                    "progress": int(progress),
                    "current_frame": i + 1,
                    "total_frames": total_frames
                }
                
                caption = generate_caption_with_beam_search(frame, is_video_frame=True)
                if caption and caption != "Unable to generate caption.":
                    frame_captions.append(caption)
                    logger.info(f"Frame {i+1}/{total_frames} caption: {caption}")
            except Exception as e:
                logger.warning(f"Failed to caption frame {i+1}/{total_frames}: {e}")
                continue
        
        processing_status[task_id] = {"status": "finalizing", "progress": 85}
        
        # Process captions to create video description
        if frame_captions:
            # Remove duplicates while preserving order
            unique_captions = []
            seen = set()
            for caption in frame_captions:
                caption_lower = caption.lower()
                is_duplicate = False
                for seen_caption in seen:
                    if (caption_lower in seen_caption or seen_caption in caption_lower or 
                        len(set(caption_lower.split()) & set(seen_caption.split())) > len(caption_lower.split()) * 0.7):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_captions.append(caption)
                    seen.add(caption_lower)
            
            # Create comprehensive video description
            if len(unique_captions) == 1:
                final_caption = unique_captions[0]
            elif len(unique_captions) <= 2:
                final_caption = ". ".join(unique_captions)
            elif len(unique_captions) <= 4:
                final_caption = f"{unique_captions[0]}. The video also shows {unique_captions[1].lower()}"
                if len(unique_captions) > 2:
                    final_caption += f" and {unique_captions[-1].lower()}"
            else:
                beginning = unique_captions[0]
                middle = unique_captions[len(unique_captions)//2]
                end = unique_captions[-1]
                final_caption = f"The video begins with {beginning.lower()}, continues to show {middle.lower()}, and ends with {end.lower()}"
        else:
            final_caption = "Unable to generate video description."
        
        processing_status[task_id] = {
            "status": "completed", 
            "progress": 100, 
            "caption": final_caption,
            "frames_processed": len(frames),
            "captions_generated": len(frame_captions)
        }
        
        return {"caption": final_caption, "frames_processed": len(frames), "captions_generated": len(frame_captions)}
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}", exc_info=True)
        processing_status[task_id] = {"status": "error", "message": str(e)}
        return {"caption": "Video processing failed.", "error": str(e)}

@app.post("/caption-video/")
async def caption_video(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    temp_path = f"temp_{task_id}.mp4"
    
    try:
        processing_status[task_id] = {"status": "uploading", "progress": 5}
        
        # Save uploaded file
        file_content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Start processing in background
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(executor, process_video_frames, temp_path, task_id)
        
        # Return task ID for progress tracking
        return {"task_id": task_id, "status": "processing"}
        
    except Exception as e:
        logger.error(f"Video upload failed: {str(e)}")
        processing_status[task_id] = {"status": "error", "message": str(e)}
        return {"task_id": task_id, "status": "error", "message": str(e)}

@app.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """Get the status of video processing."""
    if task_id not in processing_status:
        return {"status": "not_found"}
    
    status = processing_status[task_id].copy()
    
    # Clean up completed or errored tasks after returning status
    if status.get("status") in ["completed", "error"]:
        # Schedule cleanup
        def cleanup():
            if task_id in processing_status:
                del processing_status[task_id]
            temp_path = f"temp_{task_id}.mp4"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")
        
        # Clean up after a delay
        threading.Timer(5.0, cleanup).start()
    
    return status

@app.post("/caption-live/")
async def caption_live(file: UploadFile = File(...)):
    global speak_enabled
    if not speak_enabled:
        return {"caption": ""}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Use conditional generation for live captions
        prompt = "A photo of"  # Helps with context
        caption = generate_caption_with_beam_search(image, prompt)
        
        logger.info(f"Live caption: {caption}")
        return {"caption": caption}
        
    except Exception as e:
        logger.error(f"Live captioning failed: {e}")
        return {"caption": "Live caption failed."}

@app.post("/stop-speaking/")
async def stop_speaking():
    global speak_enabled
    speak_enabled = False
    logger.info("Speech stopped")
    return JSONResponse({"message": "Speech stopped."})

@app.post("/resume-speaking/")
async def resume_speaking():
    global speak_enabled
    speak_enabled = True
    logger.info("Speech resumed")
    return JSONResponse({"message": "Speech resumed."})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }
