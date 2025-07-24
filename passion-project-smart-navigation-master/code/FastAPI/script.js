let liveStream = null;
let liveInterval = null;
let isLiveActive = false;
let abortController = null;
let isSpeaking = false;
let speechQueue = [];
let lastSpokenCaption = "";

// Enhanced image preview with validation
document.getElementById('img-input').onchange = e => {
  const file = e.target.files[0];
  if (file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Image file is too large. Please select a file smaller than 10MB.');
      return;
    }
    
    document.getElementById('img-preview').src = URL.createObjectURL(file);
    document.getElementById('img-preview-container').classList.add('show');
    document.getElementById('img-caption-btn').disabled = false;
  }
};

// Enhanced video preview with validation
document.getElementById('vid-input').onchange = e => {
  const file = e.target.files[0];
  if (file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
      alert('Please select a valid video file.');
      return;
    }
    
    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      alert('Video file is too large. Please select a file smaller than 100MB.');
      return;
    }
    
    document.getElementById('vid-preview').src = URL.createObjectURL(file);
    document.getElementById('vid-preview-container').classList.add('show');
    document.getElementById('vid-caption-btn').disabled = false;
  }
};

// Enhanced speech synthesis with queue management for live captioning
const speakCaption = (id, isLive = false) => {
  const text = document.getElementById(id).textContent;
  if ('speechSynthesis' in window && text) {
    if (isLive) {
      // For live captioning, use the queue system
      speakLiveCaption(text);
    } else {
      // For regular captioning, speak immediately
      window.speechSynthesis.cancel(); // stop any ongoing
      speakText(text);
    }
  }
};

// Queue-based speech for live captioning
const speakLiveCaption = (text) => {
  // Skip if caption is too short or same as last spoken
  if (text.length < 3 || text === lastSpokenCaption) {
    return;
  }
  
  // If currently speaking, add to queue
  if (isSpeaking) {
    speechQueue.push(text);
    // Keep queue size manageable
    if (speechQueue.length > 3) {
      speechQueue.shift(); // Remove oldest
    }
    return;
  }
  
  // Speak immediately
  speakText(text, true);
  lastSpokenCaption = text;
};

// Core speech function
const speakText = (text, isLive = false) => {
  if (!text) return;
  
  isSpeaking = true;
  
  const utterance = new SpeechSynthesisUtterance(text);
  
  // Try to use a better voice if available
  const voices = window.speechSynthesis.getVoices();
  const preferredVoice = voices.find(voice => 
    voice.lang.startsWith('en') && 
    (voice.name.includes('Google') || voice.name.includes('Microsoft'))
  );
  
  if (preferredVoice) {
    utterance.voice = preferredVoice;
  }
  
  utterance.rate = isLive ? 1.0 : 0.9; // Slightly faster for live
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  
  utterance.onend = () => {
    isSpeaking = false;
    
    // Process queue if there are items
    if (speechQueue.length > 0 && isLive) {
      const nextText = speechQueue.shift();
      setTimeout(() => speakText(nextText, true), 500); // Small delay between speeches
    }
  };
  
  utterance.onerror = () => {
    isSpeaking = false;
    console.error('Speech synthesis error');
  };
  
  window.speechSynthesis.speak(utterance);
};

// Enhanced image captioning with progress indication
async function captionImage() {
  const file = document.getElementById('img-input').files[0];
  if (!file) return;

  const btn = document.getElementById('img-caption-btn');
  const text = document.getElementById('img-caption-text');
  const result = document.getElementById('img-caption-result');

  btn.innerHTML = '<span class="loading-dots">Analyzing image</span>';
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    const res = await fetch("/caption-image/", { 
      method: "POST", 
      body: formData,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    const data = await res.json();

    if (data.caption && data.caption !== "Image processing failed.") {
      text.textContent = data.caption;
      result.classList.add("show");
      speakCaption("img-caption-text");
    } else {
      text.textContent = "Unable to generate caption for this image.";
      result.classList.add("show");
    }

  } catch (error) {
    console.error('Caption error:', error);
    text.textContent = error.name === 'AbortError' ? 
      "Request timed out. Please try again." : 
      "Error processing image. Please try again.";
    result.classList.add("show");
  } finally {
    btn.innerHTML = "Generate Caption";
    btn.disabled = false;
  }
}

// Enhanced video captioning with progress tracking
async function captionVideo() {
  const file = document.getElementById('vid-input').files[0];
  if (!file) return;

  const btn = document.getElementById('vid-caption-btn');
  const text = document.getElementById('vid-caption-text');
  const result = document.getElementById('vid-caption-result');

  btn.innerHTML = '<span class="loading-dots">Starting video analysis</span>';
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", file);

    // Start video processing
    const res = await fetch("/caption-video/", { 
      method: "POST", 
      body: formData
    });
    
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    
    const data = await res.json();
    
    if (data.status === "error") {
      throw new Error(data.message || "Video processing failed");
    }
    
    const taskId = data.task_id;
    
    // Poll for progress
    await pollVideoProgress(taskId, btn, text, result);

  } catch (error) {
    console.error('Video caption error:', error);
    text.textContent = error.message || "Error processing video. Please try again.";
    result.classList.add("show");
    btn.innerHTML = "Generate Video Narration";
    btn.disabled = false;
  }
}

// Function to poll video processing progress
async function pollVideoProgress(taskId, btn, text, result) {
  const maxPolls = 120; // 10 minutes max (5 second intervals)
  let pollCount = 0;
  
  const poll = async () => {
    try {
      const response = await fetch(`/video-status/${taskId}`);
      const status = await response.json();
      
      pollCount++;
      
      switch (status.status) {
        case 'uploading':
          btn.innerHTML = '<span class="loading-dots">Uploading video</span>';
          break;
          
        case 'extracting_frames':
          btn.innerHTML = '<span class="loading-dots">Extracting frames</span>';
          break;
          
        case 'processing_frames':
          const frameProgress = status.current_frame && status.total_frames ? 
            ` (${status.current_frame}/${status.total_frames})` : '';
          btn.innerHTML = `<span class="loading-dots">Processing frames${frameProgress}</span>`;
          break;
          
        case 'finalizing':
          btn.innerHTML = '<span class="loading-dots">Finalizing description</span>';
          break;
          
        case 'completed':
          if (status.caption && status.caption !== "Video processing failed.") {
            text.textContent = status.caption;
            result.classList.add("show");
            speakCaption("vid-caption-text");
          } else {
            text.textContent = "Unable to generate video description.";
            result.classList.add("show");
          }
          btn.innerHTML = "Generate Video Narration";
          btn.disabled = false;
          return; // Stop polling
          
        case 'error':
          throw new Error(status.message || "Video processing failed");
          
        case 'not_found':
          throw new Error("Video processing task not found");
          
        default:
          if (pollCount >= maxPolls) {
            throw new Error("Video processing timed out");
          }
          break;
      }
      
      // Continue polling if not completed or errored
      if (status.status !== 'completed' && status.status !== 'error' && pollCount < maxPolls) {
        setTimeout(poll, 5000); // Poll every 5 seconds
      } else if (pollCount >= maxPolls) {
        throw new Error("Video processing timed out");
      }
      
    } catch (error) {
      console.error('Progress polling error:', error);
      text.textContent = error.message || "Error checking video processing status.";
      result.classList.add("show");
      btn.innerHTML = "Generate Video Narration";
      btn.disabled = false;
    }
  };
  
  // Start polling
  setTimeout(poll, 2000); // First poll after 2 seconds
}

// Enhanced live frame sending with better error handling
async function sendLiveFrame(video, textElem, resultElem) {
  if (!isLiveActive) {
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!isLiveActive) {
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const res = await fetch("/caption-live/", {
        method: "POST",
        body: formData,
        signal: abortController?.signal
      });
      
      if (!isLiveActive) {
        return;
      }
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.caption && data.caption.trim()) {
        textElem.textContent = data.caption;
        resultElem.classList.add("show");
        
        // Always attempt to speak live captions if they're meaningful
        if (data.caption.length > 3) {
          speakCaption("live-caption-text", true); // Pass true for live mode
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError' && isLiveActive) {
        console.error("Live caption failed:", err);
        textElem.textContent = "Live captioning temporarily unavailable.";
      }
    }
  }, "image/jpeg", 0.8); // Reduce quality for faster upload
}

// Enhanced live captioning with better camera handling
async function startLive() {
  const video = document.getElementById("live-video");
  const textElem = document.getElementById("live-caption-text");
  const resultElem = document.getElementById("live-caption-result");
  const container = document.getElementById("live-preview-container");
  const startBtn = document.getElementById("start-live-btn");
  const stopBtn = document.getElementById("stop-live-btn");
  const stopSpeechBtn = document.getElementById("stop-speech-btn");

  try {
    isLiveActive = true;
    abortController = new AbortController();
    
    // Reset speech state
    isSpeaking = false;
    speechQueue = [];
    lastSpokenCaption = "";
    
    // Request camera with preferred settings
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 15 }
      }
    };
    
    liveStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = liveStream;
    await video.play();
    
    container.classList.add("show");
    
    startBtn.disabled = true;
    stopBtn.disabled = false;
    stopSpeechBtn.disabled = false;

    // Initial caption after video stabilizes
    setTimeout(() => {
      if (isLiveActive) {
        sendLiveFrame(video, textElem, resultElem);
      }
    }, 1000);

    // Regular captioning with optimized interval
    liveInterval = setInterval(() => {
      if (isLiveActive) {
        sendLiveFrame(video, textElem, resultElem);
      }
    }, 6000); // Reduced to 6 seconds for better responsiveness

  } catch (err) {
    isLiveActive = false;
    console.error('Camera access error:', err);
    
    let errorMessage = "Unable to access camera. ";
    if (err.name === 'NotAllowedError') {
      errorMessage += "Please allow camera access and try again.";
    } else if (err.name === 'NotFoundError') {
      errorMessage += "No camera found on this device.";
    } else {
      errorMessage += "Please check your camera settings.";
    }
    
    alert(errorMessage);
  }
}

// Enhanced stop function with proper cleanup
function stopLive() {
  const video = document.getElementById("live-video");
  const startBtn = document.getElementById("start-live-btn");
  const stopBtn = document.getElementById("stop-live-btn");
  const stopSpeechBtn = document.getElementById("stop-speech-btn");

  isLiveActive = false;
  
  if (abortController) {
    abortController.abort();
    abortController = null;
  }

  if (liveStream) {
    liveStream.getTracks().forEach(track => {
      track.stop();
      console.log(`Stopped ${track.kind} track`);
    });
    liveStream = null;
  }

  if (liveInterval) {
    clearInterval(liveInterval);
    liveInterval = null;
  }

  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  stopSpeechBtn.disabled = true;
  
  // Clean up speech state
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  isSpeaking = false;
  speechQueue = [];
  lastSpokenCaption = "";
  
  console.log("Live captioning stopped");
}

// Enhanced speech control
async function stopSpeech() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  
  // Clear speech state
  isSpeaking = false;
  speechQueue = [];
  
  try {
    await fetch("/stop-speaking/", { 
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (err) {
    console.error("Failed to stop backend speech:", err);
  }
}

// Resume speech function
async function resumeSpeech() {
  try {
    await fetch("/resume-speaking/", { 
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (err) {
    console.error("Failed to resume backend speech:", err);
  }
}

// Add loading animation CSS
const style = document.createElement('style');
style.textContent = `
  .loading-dots:after {
    content: '...';
    animation: loading-animation 1.5s infinite;
  }
  
  @keyframes loading-animation {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
  }
`;
document.head.appendChild(style);

// Initialize speech synthesis voices
if ('speechSynthesis' in window) {
  // Load voices
  speechSynthesis.getVoices();
  
  // Handle voice loading for some browsers
  speechSynthesis.onvoiceschanged = () => {
    speechSynthesis.getVoices();
  };
}
