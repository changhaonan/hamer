import cv2
import os
import re
from pathlib import Path

def natural_sort_key(s):
    # Extract the frame number from the filename
    match = re.search(r'frame_(\d+)', s)
    if match:
        return int(match.group(1))
    return 0

def create_video_from_images(image_dir, output_video_path, fps=30):
    # Get all image files and sort them naturally
    image_files = [f for f in os.listdir(image_dir) if f.startswith('frame_') and f.endswith('_hand_keypoints_hammer.jpg')]
    image_files.sort(key=natural_sort_key)
    
    if not image_files:
        print("No matching images found!")
        return
    
    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
            print(f"Added frame: {image_file}")
        else:
            print(f"Failed to read image: {image_file}")
    
    # Release the video writer
    video.release()
    print(f"Video created successfully at: {output_video_path}")

if __name__ == "__main__":
    # Set the input directory and output video path
    image_dir = "/home/haonan/Project/hamer/example_data/test_008/handtrack/debug_images"
    output_video = "output_video.mp4"
    
    # Create the video
    create_video_from_images(image_dir, output_video) 