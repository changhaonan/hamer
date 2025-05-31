import cv2
import numpy as np
from ultralytics import YOLO
import torch

def detect_hand_keypoints(image_path):
    # Load YOLOv8 model
    model = YOLO('/home/haonan/Project/yolo-hand-pose/model/best.pt')  # Load the pose estimation model
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Run YOLOv8 inference on the image
    results = model(image)
    
    # Visualize the results on the image
    annotated_image = results[0].plot()
    
    # Display the annotated image
    cv2.imshow("Hand Keypoint Detection", annotated_image)
    
    # Save the result
    output_path = image_path.replace('.png', '_yolo.png')
    cv2.imwrite(output_path, annotated_image)
    print(f"Result saved as {output_path}")
    
    # Wait for a key press and then close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'your_image.jpg' with your image path
    image_path = '/home/haonan/Project/hamer/example_data/test_image/test.png'
    detect_hand_keypoints(image_path)
