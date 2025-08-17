from contextlib import asynccontextmanager
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import cv2
import time
import torch
import base64







midas = None
transform = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global midas, transform, device
    print("Loading MiDaS model...")
    try:
        model_type = "MiDaS_small"
        
        # Load the main model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Load the transformation function
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform

        # Validation to ensure models loaded
        if midas is None or transform is None:
            raise RuntimeError("Failed to load MiDaS model or transforms from torch.hub.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)
        midas.eval()
        
        print(f"MiDaS model loaded successfully on device: {device}")
    
    except Exception as e:
        print(f"FATAL: Could not load model during startup: {e}")
        raise e
    
    yield
    print("Cleaning up model.")
    midas = None
    transform = None
    device = None

app = FastAPI(lifespan=lifespan)

OBSTACLE_THRESHOLD = 250
DEBOUNCE_FRAMES = 5 
INSTRUCTION_COOLDOWN_SECONDS = 2.0 

class Navigation():

    def __init__(self):
        self.last_given_instruction = None
        self.last_instruction_time = 0
        self.stable_instruction = None
        self.debounce_counter = 0

def process_frame_for_navigation(img_rgb,nav):


# Depth Prediction
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    #  Image Segmentation
    img_width = img_rgb.shape[1]
    segments = {
        "Left": (0, int(0.33 * img_width)),
        "Center": (int(0.33 * img_width), int(0.66 * img_width)),
        "Right": (int(0.66 * img_width), img_width)
    }

    # Analyze Segments
    segment_avg_depths = {}
    for name, (start, end) in segments.items():
        segment_depth = depth_map[:, start:end]
        avg_depth = np.mean(segment_depth)
        segment_avg_depths[name] = avg_depth


    clearest_segment_name = min(segment_avg_depths, key=segment_avg_depths.get)
    clearest_segment_depth_value = segment_avg_depths[clearest_segment_name]


    if clearest_segment_depth_value > OBSTACLE_THRESHOLD:
        current_instruction = "Stop! Obstacle ahead."
    else:
        if clearest_segment_name == "Center":
            current_instruction = "Go Straight."
        elif clearest_segment_name == "Left":
            current_instruction = "Go Left side"
        elif clearest_segment_name == "Right":
            current_instruction = "Go Right side"

    # --- Debounce Logic ---
    if current_instruction == nav.stable_instruction:
        nav.debounce_counter += 1
    else:
        nav.stable_instruction = current_instruction
        nav.debounce_counter = 1

    # --- Cooldown and Instruction Logic ---
    current_time = time.time()
    if (nav.debounce_counter >= DEBOUNCE_FRAMES and
            nav.stable_instruction != nav.last_given_instruction and
            current_time - nav.last_instruction_time > INSTRUCTION_COOLDOWN_SECONDS):
        
        nav.last_given_instruction = nav.stable_instruction
        nav.last_instruction_time = current_time
        return nav.last_given_instruction
    return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("Client connected.")

    nav = Navigation()

    try:
        while True:
            data = await websocket.receive_text()

            img_data = base64.b64decode(data)

            np_arr = np.frombuffer(img_data,np.uint8)

            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
                instruction = process_frame_for_navigation(img_rgb, nav)

                if instruction:
                    await websocket.send_text(instruction)
                    print(f"Sent instruction: {instruction}")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011)

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running. Connect to /ws for WebSocket."}
    
