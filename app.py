import io
import time
import tempfile
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


from model import HazardPredictionModel


st.set_page_config(page_title="DHPR | Hazard Intelligence", page_icon="🛡️", layout="wide")


st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
        color: #24292e;
    }
    /* Headers */
    h1, h2, h3 {
        color: #0366d6 !important;
        font-family: 'Inter', sans-serif;
    }
    /* primary styling */
    .stButton > button {
        background: linear-gradient(90deg, #0366d6, #005cc5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(3, 102, 214, 0.3);
        color: white;
    }
    .hazard-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #d73a49;
        background: #ffeef0;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #d73a49;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    .frame-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e1e4e8;
    }
    .status-text {
        color: #586069;
        font-size: 0.9rem;
        margin-top: 15px;
        border-top: 1px solid #e1e4e8;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Cache the heavy ML model so it's not reloaded on every run."""
    return HazardPredictionModel(device="cpu")


def extract_video_frames(video_bytes: bytes, max_frames=24) -> list[Image.Image]:
    """Extract up to `max_frames` evenly spaced from the video to create a smooth output without crashing CPU."""
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return frames
            
        
        step = max(1, total_frames // max_frames)
        
        for i in range(1, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= max_frames:
                break
        cap.release()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return frames


def draw_bounding_boxes(image: Image.Image, boxes_info: list) -> Image.Image:
    """Draw real YOLO boxes with highly legible labels onto the image."""
    display_img = image.copy()
    draw = ImageDraw.Draw(display_img, "RGBA")
    colors = [(215, 58, 73), (3, 102, 214), (40, 167, 69), (255, 153, 0), (111, 66, 193)]
    
    
    try:
        font = ImageFont.load_default(size=20)
    except:
        font = ImageFont.load_default()
    
    for i, box in enumerate(boxes_info):
        x1, y1, x2, y2 = box["coords"]
        label = box["label"].upper()
        conf = box["confidence"]
        
        color = colors[i % len(colors)]
        fill_color = color + (40,) 
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill_color, width=4)
        
        text = f" {label} {conf:.2f} "
        
        
        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(text) * 12, 20
            
        
        rect_y1 = max(0, y1 - th - 8)
        draw.rectangle([x1, rect_y1, x1 + tw + 8, y1], fill=color)
        
        
        draw.text((x1 + 4, rect_y1 + 2), text, fill=(255, 255, 255), font=font)
        
    return display_img


IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]

st.title("🛡️ Driving Hazard Prediction (Real VLM)")
st.markdown("---")

# Pre-load the ML models early so analysis doesn't feel "very slow" later
with st.spinner("Warming up AI Models (this runs once to ensure instant analysis later)..."):
    _ = load_model()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Upload Feed")
    uploaded_file = st.file_uploader(
        "Drop an image or video here",
        type=IMAGE_TYPES + VIDEO_TYPES
    )

    is_video = False
    input_frames = []

    if uploaded_file is not None:
        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        is_video = file_ext in VIDEO_TYPES

        if is_video:
            st.video(uploaded_file)
            uploaded_file.seek(0)
            with st.spinner("Extracting timeline frames for processing..."):
                input_frames = extract_video_frames(uploaded_file.read(), max_frames=24)
            if input_frames:
                st.success(f"✅ Extracted {len(input_frames)} frames for video processing")
            else:
                st.error("❌ Could not extract frames from video.")
        else:
            uploaded_file.seek(0)
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Source Image", use_container_width=True)
            input_frames = [img]

        st.info("💡 Note: Running the actual ML models on CPU will take time (~5s per frame).")
        analyze_btn = st.button("🔍 Run Full Hazard Analysis", type="primary", use_container_width=True)

with col2:
    st.subheader("2. AI Analysis stream")

    if uploaded_file is not None and input_frames:
        if analyze_btn:
            with st.spinner("Initializing Hazard Prediction Engine..."):
                start_t = time.time()
                model = load_model()
            
            total = len(input_frames)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            
            heavy_model_intervals = max(1, total // 5)
            current_caption = ""
            
            st.info("💡 Compiling video output... Running fast object detection on all frames and heavy context analysis on keyframes.")
            
            for i, frame in enumerate(input_frames):
                status_text.text(f"Analyzing frame {i+1} of {total}...")
                
                t0 = time.time()
                
                if not is_video or i % heavy_model_intervals == 0 or i == 0:
                    caption, boxes = model.generate(frame)
                    current_caption = caption
                else:
                    
                    boxes = model.get_bounding_boxes(frame)
                    caption = current_caption
                    
                t1 = time.time()
                disp_frame = draw_bounding_boxes(frame, boxes)
                
                
                draw = ImageDraw.Draw(disp_frame)
                
                risk_level = caption.split("\\n")[0][:40] 
                draw.rectangle([0, 0, 400, 30], fill=(0,0,0,180))
                draw.text((10, 5), risk_level.replace("**", ""), fill=(255,255,255))
                
                results.append({
                    "frame_idx": i,
                    "caption": caption,
                    "boxes": boxes,
                    "image": disp_frame,
                    "time": t1 - t0,
                    "is_keyframe": (not is_video or i % heavy_model_intervals == 0 or i == 0)
                })
                progress_bar.progress((i + 1) / total)
                
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Analysis complete! Total time: {time.time()-start_t:.1f}s")
            
            if not is_video:
                # Image view
                res = results[0]
                st.caption(f"⚡ Processing time: {res['time']:.2f}s")
                st.markdown('#### 🧠 AI Hazard Prediction')
                st.markdown(f'<div class="hazard-text">\\n{res["caption"]}\\n</div>', unsafe_allow_html=True)
                
                st.markdown('#### 👀 Detected Objects (What the AI sees)')
                st.image(res['image'], use_container_width=True)
                
                with st.expander("Show Raw YOLO Detections"):
                    st.json(res["boxes"])
            else:
                
                st.markdown("### 🎥 Processed Output Video")
                with st.spinner("Compiling animated result..."):
                    gif_path = os.path.join(tempfile.gettempdir(), "output_dhpr.gif")
                    images = [r["image"] for r in results]
                    
                    images[0].save(
                        gif_path, 
                        save_all=True, 
                        append_images=images[1:], 
                        optimize=False, 
                        duration=200, 
                        loop=0
                    )
                    
                st.image(gif_path, use_container_width=True)
                
                st.markdown("### ⏱️ Keyframe Breakdown")
                
                for res in results:
                    if res.get("is_keyframe"):
                        with st.container():
                            st.markdown(f'<div class="frame-container">', unsafe_allow_html=True)
                            st.markdown(f"**Keyframe {res['frame_idx']+1}** • (Processing time: {res['time']:.2f}s)")
                            st.markdown('**🧠 AI Hazard Prediction:**')
                            st.markdown(f'<div class="hazard-text">\\n{res["caption"]}\\n</div>', unsafe_allow_html=True)
                            st.markdown('**👀 Detected Objects in Scene:**')
                            st.image(res['image'], use_container_width=True)
                            if res["boxes"]:
                                with st.expander(f"View Raw Detections for Frame {res['frame_idx']+1}"):
                                    st.write(", ".join([f"{b['label']} ({b['confidence']*100:.0f}%)" for b in res["boxes"]]))
                            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting input. Upload an image or video to begin.")
