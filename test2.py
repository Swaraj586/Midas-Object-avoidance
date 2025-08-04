import cv2
import torch
import numpy as np
import time

print("PyTorch version:", torch.__version__)

# Model
model_type = "MiDaS_small"
print("Loading MiDaS model:", model_type)
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


cap = cv2.VideoCapture(1)

#  Navigation
last_given_instruction = None
instruction_cooldown_seconds = 5.0
last_instruction_time = 0

stable_instruction = None
debounce_counter = 0
DEBOUNCE_FRAMES = 10


OBSTACLE_THRESHOLD = 250

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to grab frame")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    if current_instruction == stable_instruction:
        debounce_counter += 1
    else:
        stable_instruction = current_instruction
        debounce_counter = 1

    # --- Cooldown and Instruction Logic ---
    current_time = time.time()
    if (debounce_counter >= DEBOUNCE_FRAMES and
            stable_instruction != last_given_instruction and
            current_time - last_instruction_time > instruction_cooldown_seconds):
        print(f"--- NEW INSTRUCTION: {stable_instruction} ---")
        last_given_instruction = stable_instruction
        last_instruction_time = current_time

    # --- Visualization (Optional) ---
    # Highlight the path that is being recommended or stop if blocked
    if current_instruction == "Stop! Obstacle ahead.":
        # Highlight the whole screen red to indicate stop
        highlight_color = (0, 0, 255)  # Red for stop
        start, end = (0, 0), (img.shape[1], img.shape[0])
        cv2.rectangle(img, start, end, highlight_color, 5)  # Draw a border
    else:
        # Highlight the clearest path green
        start, end = segments[clearest_segment_name]
        overlay = img.copy()
        cv2.rectangle(overlay, (start, 0), (end, img.shape[0]), (0, 255, 0), -1)  # Green for go
        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


    cv2.putText(img, f"Instruction: {last_given_instruction or '...'}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    output = cv2.resize(img, (960, 540))
    cv2.imshow("Output", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()