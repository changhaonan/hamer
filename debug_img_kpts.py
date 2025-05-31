import cv2
from typing import List, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

def draw_hand_landmarks(
    image: Image, landmarks: List[Tuple[int, int]], connections: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2, use_dash: bool = False
) -> np.ndarray:
    """
    Draw hand landmarks and connections on an image.

    Args:
        image: Input image
        landmarks: List of (x, y) coordinates for each landmark
        connections: List of (start_idx, end_idx) pairs defining connections between landmarks
        color: RGB color for drawing (default: green)
        thickness: Line thickness for drawing
        use_dash: Whether to draw dashed lines (default: False)

    Returns:
        Image with hand landmarks and connections drawn as numpy array
    """

    # Validate inputs
    if not landmarks or not connections:
        return image

    # Validate landmark indices in connections
    max_idx = len(landmarks) - 1
    valid_connections = [(start, end) for start, end in connections if 0 <= start <= max_idx and 0 <= end <= max_idx]

    # Draw connections
    for start_idx, end_idx in valid_connections:
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        if use_dash:
            # Calculate line length and direction
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            # Number of dashes
            num_dashes = int(length / 30)  # Adjust dash length by changing divisor
            
            if num_dashes > 0:
                # Calculate dash and gap lengths
                dash_length = length / (2 * num_dashes)
                
                # Draw dashes
                for i in range(num_dashes):
                    # Calculate start and end points for this dash
                    dash_start_x = start_point[0] + (2 * i * dash_length * dx / length)
                    dash_start_y = start_point[1] + (2 * i * dash_length * dy / length)
                    dash_end_x = start_point[0] + ((2 * i + 1) * dash_length * dx / length)
                    dash_end_y = start_point[1] + ((2 * i + 1) * dash_length * dy / length)
                    
                    # Draw this dash
                    cv2.line(image, 
                            (int(dash_start_x), int(dash_start_y)),
                            (int(dash_end_x), int(dash_end_y)),
                            color, thickness, cv2.LINE_AA)
        else:
            cv2.line(image, start_point, end_point, color, thickness, cv2.LINE_AA)

    # Draw landmarks
    for x, y in landmarks:
        cv2.circle(image, (x, y), 3, color, -1)

    return image


def visualize_landmarks(video_path: str, landmarks: List[List[List[Tuple[int, int]]]], handedness: List[str], confidence: List[float], output_path: str):
    """
    Visualize landmarks on the video.
    
    Args:
        video_path (str): Path to input video file
        landmarks (List[List[List[Tuple[int, int]]]]): List of frame landmarks, where each frame contains a list of hand landmarks
        handedness (List[str]): List of handedness for each frame
        output_path (str): Path to save output video file
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define colors for each hand type
    colors = {
        "left": (0, 0, 255),  # Red for left hand
        "right": (0, 255, 0)  # Green for right hand
    }
    
    # Define connections between landmarks (MediaPipe hand connections)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 5), (5, 9), (9, 13), (13, 17),  # palm connections
    ]
    
    frame_idx = 0
    count_low_confidence = 0
    count_detected = 0
    pbar = tqdm(total=len(landmarks))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Get landmarks for current frame
        frame_landmarks = landmarks[frame_idx] if frame_idx < len(landmarks) else None
        frame_handedness = handedness[frame_idx] if frame_idx < len(handedness) else None
        frame_confidence = confidence[frame_idx] if frame_idx < len(confidence) else None
        
        if frame_landmarks and frame_handedness:
            # Draw landmarks for each hand
            for hand_landmarks, hand_type, _confidence in zip(frame_landmarks, frame_handedness, frame_confidence):
                color = colors.get(hand_type.lower(), (255, 255, 255))  # Default to white if hand type not found
                use_dash = _confidence is not None and _confidence < 0.5
                frame = draw_hand_landmarks(frame, hand_landmarks, connections, color, 2, use_dash)
            if len(frame_confidence) == 0 or min(frame_confidence) < 0.5:
                count_low_confidence += 1
            count_detected += 1
        # Write frame to output video
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    # Print statistics
    print(f"Total frames: {frame_idx}")
    print(f"Detected frames: {count_detected}")
    print(f"Low confidence frames: {count_low_confidence}, take {count_low_confidence/frame_idx*100}%")
    # Release resources
    cap.release()
    out.release()
    print(f"Visualization complete. Output saved to {output_path}")


if __name__ == "__main__":
    video_path = "/home/haonan/Project/hamer/example_data/test_001/camera_0.mp4"
    landmarks_path = "/home/haonan/Project/hamer/demo_out/exp_3/camera_0_hammer_landmarks.json"
    output_path = "/home/haonan/Project/hamer/example_data/test_001/camera_0_hammer_test.mp4"
    with open(landmarks_path, "r") as f:
        landmark_data = json.load(f)
    landmarks = [frame["landmarks"] for frame in landmark_data]
    handedness = [frame["handedness"] for frame in landmark_data]
    confidence = [frame["confidence"] for frame in landmark_data]
    visualize_landmarks(video_path=video_path, landmarks=landmarks, handedness=handedness, confidence=confidence, output_path=output_path)