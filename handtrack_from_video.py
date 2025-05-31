"""
This script is used to detect the hand keypoints from the image.

Usage:
python demo_img_kpts.py --checkpoint /path/to/checkpoint --video_path /path/to/video --out_path /path/to/output

"""
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import time  # Add time module import
from typing import List, Tuple
import json
import mediapipe as mp
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD

from vitpose_model import ViTPoseModel

# Define hand connections (same as in hand_track.py)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),  # palm connections
]

def recovery_kpts(kpts_2d, bboxes, img_size, is_right):
    """Recovery the keypoints from the image size to the original image size
    
    Args:
        kpts_2d: Keypoints in normalized coordinates [-0.5, 0.5], shape (N, K, 2) where N is batch size
        bboxes: List of bounding boxes [x1, y1, x2, y2], shape (N, 4)
        img_size: Original image size [width, height]
        is_right: list[bool]: Whether this is a right hand, shape (N,)
    
    Returns:
        Keypoints in original image coordinates, shape (N, K, 2)
    """
    # Calculate center and scale from bbox for each item in batch
    center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0  # shape (N, 2)

    # Calculate bbox size for each item
    bbox_size = bboxes[:, 2:4] - bboxes[:, 0:2]
    
    # Convert normalized coordinates to pixel coordinates
    kpts_2d = kpts_2d.copy()
    # Handle flipping for left hands
    right_mask = is_right.astype(bool)
    if np.any(np.logical_not(right_mask)):
        kpts_2d[np.logical_not(right_mask), :, 0] = - kpts_2d[np.logical_not(right_mask), :, 0]
    kpts_2d[..., [0]] = (kpts_2d[..., [0]]) * bbox_size[:, None, [0]] + center[:, None, [0]]
    kpts_2d[..., [1]] = (kpts_2d[..., [1]]) * bbox_size[:, None, [1]] + center[:, None, [1]]
    
    return kpts_2d


def save_landmarks(landmarks: List[List[List[Tuple[int, int]]]], handedness: List[str],  output_path: str, confidence: List[float]=None):
    """
    Save landmarks to a JSON file.
    """
    # Save landmarks to JSON file
    # json_path = output_path.rsplit(".", 1)[0] + "_landmarks.json"
    json_path = output_path.replace(".mp4", "_landmarks.json")

    # Convert landmarks to serializable format
    serializable_landmarks = []
    if confidence is None:
        confidence = [[1.0] * len(landmark) for landmark in landmarks]
    for landmark, _handedness, _confidence in zip(landmarks, handedness, confidence):
        frame_data = {}
        if landmark is None:
            pass
        else:
            frame_data["handedness"] = ["left" if x == 0 else "right" for x in _handedness]
            frame_data["landmarks"] = []
            for hand_landmark in landmark:
                hand_data = [[int(x), int(y)] for x, y in hand_landmark]
                frame_data["landmarks"].append(hand_data)
            frame_data["confidence"] = [int(_c * 100) for _c in _confidence]
        serializable_landmarks.append(frame_data)

    with open(json_path, "w") as f:
        json.dump(serializable_landmarks, f)
    print(f"Landmarks saved to {json_path}")


def ego_human_nms(bboxes, scores, img_size, iou_threshold=0.3):
    """
    Non-maximum suppression for ego human detection. Follow the following steps:
    1. First perform NMS.
    2. If after NMS, there are more than one human detection. Then use the bbox of
    the two bboxes.
    """
    from detectron2.layers.nms import nms
    if len(bboxes) == 0:
        return bboxes, scores
    # NMS
    nms_idx = nms(torch.from_numpy(bboxes), torch.from_numpy(scores), iou_threshold)
    # If after NMS, there are more than one human detection. Then a bbox that include
    # all the bboxes.
    if len(nms_idx) > 1:
        bboxes = bboxes[nms_idx]
        bboxes_lefttop = np.min(bboxes[:, :2], axis=0)
        bboxes_rightbottom = np.max(bboxes[:, 2:], axis=0)
        bboxes = np.concatenate([bboxes_lefttop, bboxes_rightbottom], axis=0).reshape(1, 4)
        scores = np.array([np.max(scores[nms_idx])])
    else:
        bboxes = bboxes[nms_idx].reshape(1, 4)
        scores = np.array([scores[nms_idx]])
    # As this is a ego human scene, we have a minimum x and maximum x. (1/4, 3/4)
    bboxes[:, 0] = np.clip(bboxes[:, 0], a_min=None, a_max=img_size[1] * 1 / 4)
    bboxes[:, 2] = np.clip(bboxes[:, 2], a_min=img_size[1] * 3 / 4, a_max=None)
    return bboxes, scores


def hand_similarity(left_hand_landmarks, right_hand_landmarks, img_size):
    """
    Calculate the similarity between two hands. Use something like pointcloud distance.
    """
    ## 1. Calculate the bbox iou
    # left_hand_bbox = np.array(left_hand_bbox)
    # right_hand_bbox = np.array(right_hand_bbox)
    left_hand_bbox = np.array([max(0, left_hand_landmarks[:, 0].min()), max(0, left_hand_landmarks[:, 1].min()), min(img_size[0], left_hand_landmarks[:, 0].max()), min(img_size[1], left_hand_landmarks[:, 1].max())])
    right_hand_bbox = np.array([max(0, right_hand_landmarks[:, 0].min()), max(0, right_hand_landmarks[:, 1].min()), min(img_size[0], right_hand_landmarks[:, 0].max()), min(img_size[1], right_hand_landmarks[:, 1].max())])
    left_hand_bbox_area = (left_hand_bbox[2] - left_hand_bbox[0]) * (left_hand_bbox[3] - left_hand_bbox[1])
    right_hand_bbox_area = (right_hand_bbox[2] - right_hand_bbox[0]) * (right_hand_bbox[3] - right_hand_bbox[1])
    
    # Calculate intersection coordinates
    x_left = max(left_hand_bbox[0], right_hand_bbox[0])
    y_top = max(left_hand_bbox[1], right_hand_bbox[1])
    x_right = min(left_hand_bbox[2], right_hand_bbox[2])
    y_bottom = min(left_hand_bbox[3], right_hand_bbox[3])
    
    # Calculate intersection area (0 if boxes don't overlap)
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    left_hand_iou = intersection_area / left_hand_bbox_area
    right_hand_iou = intersection_area / right_hand_bbox_area
    iou = max(left_hand_iou, right_hand_iou)
    
    # Convert left hand landmarks to right hand landmarks
    # Thumb -> pinky
    # Index -> ring
    # Middle -> middle
    # Ring -> index
    # Pinky -> thumb
    # Palm -> palm
    left_hand_landmarks_converted = left_hand_landmarks.copy()
    left_hand_landmarks_converted[1:4] = left_hand_landmarks[17:20]   # Pinky -> Thumb
    left_hand_landmarks_converted[5:8] = left_hand_landmarks[13:16]   # Ring -> Index
    left_hand_landmarks_converted[13:16] = left_hand_landmarks[5:8]  # Index -> Ring
    left_hand_landmarks_converted[17:20] = left_hand_landmarks[1:4]   # Thumb -> Pinky

    ## 2. Calculate the distance between the two hands
    distance = np.abs(left_hand_landmarks_converted - right_hand_landmarks).mean()
    distance = distance / max(np.sqrt(left_hand_bbox_area), np.sqrt(right_hand_bbox_area))
    # return distance < dist_thresh and iou > iou_thresh
    return distance, iou


def hand_heurstic_filter(hand_landmarks, handedness, img_size, use_area=True, use_edge=True, use_direction=True):
    """
    Filter out the hand that are too small, too close to the edge, or not pointing to the center of the image.
    """
    filter_log = {}
    idx_exclude_thumb = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    hand_bbox = np.array([max(0, hand_landmarks[idx_exclude_thumb, 0].min()), max(0, hand_landmarks[idx_exclude_thumb, 1].min()), min(img_size[0], hand_landmarks[idx_exclude_thumb, 0].max()), min(img_size[1], hand_landmarks[idx_exclude_thumb, 1].max())])
    hand_bbox_area = (hand_bbox[2] - hand_bbox[0]) * (hand_bbox[3] - hand_bbox[1])
    hand_bbox_center = np.array([(hand_bbox[0] + hand_bbox[2]) / 2, (hand_bbox[1] + hand_bbox[3]) / 2])

    # Draw the hand direction by draw an arrow between hand root and hand direction
    hand_root = hand_landmarks[0, :2]
    hand_direction = hand_bbox_center - hand_root

    # Edge filter
    # Filter if more than 30% of the hand is outside the image
    hand_bbox_full = np.array([hand_landmarks[idx_exclude_thumb, 0].min(), hand_landmarks[idx_exclude_thumb, 1].min(), hand_landmarks[idx_exclude_thumb, 0].max(), hand_landmarks[idx_exclude_thumb, 1].max()])
    hand_bbox_full_area = (hand_bbox_full[2] - hand_bbox_full[0]) * (hand_bbox_full[3] - hand_bbox_full[1])
    edge_ratio = hand_bbox_area / hand_bbox_full_area
    if use_edge and edge_ratio < 0.7:
        # Filter out the hand that are too close to the edge
        return True, filter_log
    
    # Area filter
    if use_area and hand_bbox_full_area < 4000:
        # Filter out the hand that are too small
        return True, filter_log
    
    # Handness filter
    # If it is left hand, we will filter it if the root hand is at right and the hand direction is pointing to the left
    if use_direction and (handedness == 0 and hand_root[0] > img_size[0] * 3 / 4 and hand_direction[0] < 0):
        return True, filter_log
    # If it is right hand, we will filter it if the root hand is at left and the hand direction is pointing to the right
    if use_direction and (handedness == 1 and hand_root[0] < img_size[0] * 1 / 4 and hand_direction[0] > 0):
        return True, filter_log
    
    # Save info to log
    filter_log["hand_bbox"] = hand_bbox
    filter_log["hand_bbox_center"] = hand_bbox_center
    filter_log["hand_root"] = hand_root
    filter_log["hand_bbox_area"] = hand_bbox_area
    filter_log["handedness"] = handedness
    filter_log["hand_direction"] = hand_direction
    filter_log["hand_bbox_full_area"] = hand_bbox_full_area
    filter_log["edge_ratio"] = edge_ratio
    return False, filter_log


def single_hand_classsify_mediapipe(image_patch, mediapipe_hands):
    """
    Detect hand landmarks using mediapipe.
    """
    results = mediapipe_hands.process(image_patch)
    # Return the handedness of the hand
    if results.multi_hand_landmarks is None:
        return None
    handedness_label = results.multi_handedness[0].classification[0].label
    handedness = 0 if handedness_label == "Left" else 1
    return handedness


def process_image(model, model_cfg, detector, cpm, frame, device, output_name, debug_vis=False):
    """
    Process one image with hammer hand tracking. Generate raw data.
    Args:
        model: The model to use for hand detection.
        model_cfg: The model configuration.
        detector: The detector to use for human detection.
        cpm: The keypoint detector to use for hand detection.
        frame: The frame to process.
        device: The device to use for the model.
        output_name: The name of the output file.
        debug_vis: Whether to visualize the results.
    Returns:
        The keypoints of the hands.
    """
    ########################## Step 1: Human Detection ##########################
    # Start timing human detection
    human_det_start = time.time()
    # Detect humans in frame
    det_out = detector(frame)
    img = frame.copy()[:, :, ::-1]
    vis_img = frame.copy()  # Create a copy for visualization

    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    pred_bboxes, pred_scores = ego_human_nms(pred_bboxes, pred_scores, img_size=img.shape[:2])
    human_det_time = time.time() - human_det_start
    if debug_vis:
        print(f"Human detection time: {human_det_time:.3f} seconds")

    # Visualize human detection results
    for bbox, score in zip(pred_bboxes, pred_scores):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            f"Human: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save human detection visualization
    if debug_vis:
        output_folder = Path(output_name).parent
        output_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_name}_human_detection.jpg", vis_img)

    # Disable human detection
    # human_det_time = 0.0
    # pred_bboxes = np.array([[0, 0, img.shape[1], img.shape[0]]])
    # pred_scores = np.array([1.0])

    ########################## Step 2: Detect Human pose ##########################
    # Start timing hand detection
    hand_det_start = time.time()
    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    # Create a new copy for keypoint visualization
    kpts_vis_img = frame.copy()
    kpts_vis_img_hammer = frame.copy()

    bboxes = []
    is_right = []
    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        # Visualize left hand keypoints and connections
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keyp):
                if conf > 0.5:
                    cv2.circle(kpts_vis_img, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                if keyp[start_idx, 2] > 0.5 and keyp[end_idx, 2] > 0.5:
                    start_point = (int(keyp[start_idx, 0]), int(keyp[start_idx, 1]))
                    end_point = (int(keyp[end_idx, 0]), int(keyp[end_idx, 1]))
                    cv2.line(kpts_vis_img, start_point, end_point, (0, 0, 255), 2)

            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(0)
            # Draw left hand bbox
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(kpts_vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                kpts_vis_img,
                "Left Hand",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # Visualize right hand keypoints and connections
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keyp):
                if conf > 0.5:
                    cv2.circle(kpts_vis_img, (int(x), int(y)), 3, (255, 0, 0), -1)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                if keyp[start_idx, 2] > 0.5 and keyp[end_idx, 2] > 0.5:
                    start_point = (int(keyp[start_idx, 0]), int(keyp[start_idx, 1]))
                    end_point = (int(keyp[end_idx, 0]), int(keyp[end_idx, 1]))
                    cv2.line(kpts_vis_img, start_point, end_point, (255, 0, 0), 2)

            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(1)
            # Draw right hand bbox
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(kpts_vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                kpts_vis_img,
                "Right Hand",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    hand_det_time = time.time() - hand_det_start
    if debug_vis:
        print(f"Hand detection time: {hand_det_time:.3f} seconds")

    # Save keypoint detection visualization
    if debug_vis:   
        output_folder = Path(output_name).parent
        output_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_name}_hand_keypoints.jpg", kpts_vis_img)

    if len(bboxes) == 0:
        # No thing is detected
        return kpts_vis_img_hammer, [], []

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    ########################## Step 3: Hand Reconstruction ##########################
    # Start timing hand reconstruction
    recon_start = time.time()
    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(
        model_cfg, frame, boxes, right, rescale_factor=2.0
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

    pred_keypoints_2ds = []
    handedness = []
    image_patches = []
    hand_bboxes = []
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        bbox_center = batch["box_center"].detach().cpu().numpy()
        bbox_size = batch["box_size"].detach().cpu().numpy()
        if len(bbox_size.shape) == 1:
            bbox_size = bbox_size[:, None]
        right = batch["right"].detach().cpu().numpy()
        pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()
        # recover the keypoints to the original image size
        bbox_topleft = bbox_center - bbox_size / 2.0
        bbox = np.concatenate([bbox_topleft, bbox_topleft + bbox_size], axis=1)
        img_size = batch["img_size"].detach().cpu().numpy()
        pred_keypoints_2d = recovery_kpts(pred_keypoints_2d, bbox, img_size[0], right)
        # Add to the output
        pred_keypoints_2ds.extend(pred_keypoints_2d)
        handedness.extend(right)
        hand_bboxes.extend(bbox)
        # Get the image patch
        image_patch = batch["img"].detach().cpu().numpy().transpose(0, 2, 3, 1)
        image_mean = dataset.mean
        image_std = dataset.std
        image_patch = (image_patch * image_std[None, ...] + image_mean[None, ...]).astype(np.uint8)
        image_patches.append(image_patch)

    recon_time = time.time() - recon_start
    if debug_vis:
        print(f"Hand reconstruction time: {recon_time:.3f} seconds")

    return kpts_vis_img_hammer, pred_keypoints_2ds, handedness


def hand_filtering(kpts_vis_img_hammer, pred_keypoints_2ds, handedness, img_size, mp_hand_tracker_bk, debug_vis=False, output_name=None):
    ########################## Step 1: Hand Filtering out ##########################
    filter_start = time.time()
    ## 4.1. Filter out candidates that are not valid
    pred_keypoints_2ds_filtered = []
    handedness_filtered = []
    filter_logs = []
    
    for i in range(len(pred_keypoints_2ds)):
        to_filter, filter_log = hand_heurstic_filter(pred_keypoints_2ds[i], handedness[i], img_size[0], use_area=True, use_edge=False, use_direction=True)
        if to_filter:
            continue
        # Save filter log
        filter_logs.append(filter_log)
        pred_keypoints_2ds_filtered.append(pred_keypoints_2ds[i])
        handedness_filtered.append(handedness[i])
    pred_keypoints_2ds = pred_keypoints_2ds_filtered
    handedness = handedness_filtered

    # [DEBUG]: Visualize the filter log
    area_str = ""
    # [DEBUG]: Draw the hand bbox
    for filter_log in filter_logs:
        hand_bbox = filter_log["hand_bbox"]
        hand_bbox_center = filter_log["hand_bbox_center"]
        hand_bbox_area = filter_log["hand_bbox_full_area"]
        hand_root = filter_log["hand_root"]
        edge_ratio = filter_log["edge_ratio"]
        cv2.rectangle(kpts_vis_img_hammer, (int(hand_bbox[0]), int(hand_bbox[1])), (int(hand_bbox[2]), int(hand_bbox[3])), (255, 255, 255), 1)
        # Draw the hand direction by draw an arrow between hand root and hand direction
        hand_direction = hand_bbox_center * 2 - hand_root
        cv2.arrowedLine(kpts_vis_img_hammer, (int(hand_root[0]), int(hand_root[1])), (int(hand_direction[0]), int(hand_direction[1])), (255, 0, 0), 1)
        # Put on text at right bottom corner
        area_str += f"Hand {i} Area: {hand_bbox_area:.2f}, Center: {hand_bbox_center[0]:.2f}, {hand_bbox_center[1]:.2f}, Edge Ratio: {edge_ratio:.2f}\n"

    # Split text into lines and render each line separately
    font_scale = 0.7  # Reduced font size
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = 30  # Space between lines
    
    # Starting position for text
    y_pos = kpts_vis_img_hammer.shape[0] - 500
    
    # Render each line of text
    for line in area_str.split('\n'):
        if line:  # Only render non-empty lines
            cv2.putText(kpts_vis_img_hammer, line, (10, y_pos), font, font_scale, (0, 0, 255), font_thickness)
            y_pos += line_spacing

    ## 4.2. Calculate the similarity between the two hands if there are two hands
    if len(pred_keypoints_2ds) == 2:
        left_hand_landmarks = pred_keypoints_2ds[0]
        right_hand_landmarks = pred_keypoints_2ds[1]
        distance, iou = hand_similarity(left_hand_landmarks, right_hand_landmarks, img_size=img_size[0])
        cv2.putText(kpts_vis_img_hammer, f"IOU: {iou:.2f}, Distance: {distance:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        if distance < 0.2 or iou > 0.75:
            # Determine the handness using another hand tracker
            image_patches = np.vstack(image_patches)
            assert image_patches.shape[0] == 2, "Only two hands are detected"
            left_hand_index = handedness.index(0)
            handedness_mp = single_hand_classsify_mediapipe(image_patches[left_hand_index], mp_hand_tracker_bk)
            # Draw text at the bottom
            if handedness_mp is None:
                handedness_mp_str = "Unknown"
            else:
                handedness_mp_str = "Left" if handedness_mp == 0 else "Right"
            cv2.putText(kpts_vis_img_hammer, f"Hand is overlapping, This is {handedness_mp_str} hand", (10, kpts_vis_img_hammer.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            if handedness_mp is None:
                # Filter out the hand that we can not classify
                pred_keypoints_2ds = []
                handedness = []
            else:
                mp_choice_idx = handedness.index(handedness_mp)
                pred_keypoints_2ds = [pred_keypoints_2ds[mp_choice_idx]]
                handedness = [handedness[mp_choice_idx]]

    ## 4.3. Another round of filtering, flitering out hand that are too close to edge
    pred_keypoints_2ds_filtered = []
    handedness_filtered = []
    for i in range(len(pred_keypoints_2ds)):
        hand_landmarks = pred_keypoints_2ds[i]
        # Filter out based on heuristic
        to_filter, filter_log = hand_heurstic_filter(hand_landmarks, handedness[i], img_size[0], use_area=False, use_edge=True, use_direction=False)
        if to_filter:
            continue
        pred_keypoints_2ds_filtered.append(pred_keypoints_2ds[i])
        handedness_filtered.append(handedness[i])
    pred_keypoints_2ds = pred_keypoints_2ds_filtered
    handedness = handedness_filtered

    filter_time = time.time() - filter_start
    if debug_vis:
        print(f"Hand classification time: {filter_time:.3f} seconds")

    ## 4.4. Visualize the hand keypoints
    for pred_keypoints_2d, _handedness in zip(pred_keypoints_2ds, handedness):
        color = (
            (0, 0, 255) if _handedness == 0 else (0, 255, 0)
        )  # Red for left hand, green for right hand
        # Draw keypoints
        for j in range(pred_keypoints_2d.shape[0]):
            x, y = pred_keypoints_2d[j, :2]
            cv2.circle(kpts_vis_img_hammer, (int(x), int(y)), 3, color, -1)

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_x, start_y = pred_keypoints_2d[start_idx, :2]
            end_x, end_y = pred_keypoints_2d[end_idx, :2]
            cv2.line(kpts_vis_img_hammer, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, 2)

    if debug_vis:
        output_folder = Path(output_name).parent
        output_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_name}_hand_keypoints_hammer.jpg", kpts_vis_img_hammer)

    return kpts_vis_img_hammer, pred_keypoints_2ds, handedness


def raw_process():
    parser = argparse.ArgumentParser(description="HaMeR video demo code")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/haonan/Project/hamer/example_data/test_008/raw_data/camera/camera_0.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/haonan/Project/hamer/example_data/test_008/handtrack/camera_0_hamer.mp4",
        help="Path to output video file",
    )
    parser.add_argument(
        "--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox"
    )
    parser.add_argument(
        "--body_detector",
        type=str,
        default="regnety",
        choices=["vitdet", "regnety"],
        help="Using regnety improves runtime and reduces memory",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output video FPS. If not set, uses input video FPS",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    args = parser.parse_args()
    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if args.body_detector == "vitdet":
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = (
            Path(hamer.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == "regnety":
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Mediapipe hands
    mp_hands = mp.solutions.hands
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5
    min_tracking_confidence = 0.5
    max_hands = 1
    static_image_mode = True
    model_complexity = 1
    hand_tracker_bk = mp_hands.Hands(
        model_complexity=model_complexity,
        static_image_mode=static_image_mode,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Image processing
    # for i in range(1, 10):
    #     frame = cv2.imread(f"/home/haonan/Project/hamer/example_data/ego_test_002.png")
    #     process_image(model, model_cfg, detector, cpm, frame, device, args)
    # Read video
    frame_idx = 0
    video_path = args.video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file {video_path}")
    # Output video writer
    out_video_path = args.out_path
    # Create output folder
    output_folder = Path(out_video_path).parent
    output_folder.mkdir(parents=True, exist_ok=True)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pred_keypoints_2ds_video = []
    handedness_video = []
    # start_frame_idx = 0
    # end_frame_idx = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame_idx = 0
    end_frame_idx = 600
    # start_frame_idx = 5400
    # end_frame_idx = 5600
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start_frame_idx or frame_idx > end_frame_idx:
            frame_idx += 1
            continue
        debug_image_path = os.path.join(Path(out_video_path).parent, "debug_images", f"frame_{frame_idx}")
        kpts_vis_img_hammer, pred_keypoints_2ds_frame, handedness_frame = process_image(model, model_cfg, detector, cpm, frame, device, output_name=debug_image_path, debug_vis=True)
        out_video.write(kpts_vis_img_hammer)
        # Add to the output
        pred_keypoints_2ds_video.append(pred_keypoints_2ds_frame)
        handedness_video.append(handedness_frame)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out_video.release()
    # Save landmarks
    save_landmarks(pred_keypoints_2ds_video, handedness_video, output_path=out_video_path)


def post_process(raw_data_path, output_path):
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)
    for frame_idx, frame_data in raw_data.items():
        pred_keypoints_2ds, handedness = frame_data["landmarks"], frame_data["handedness"]
        # Perform filtering algorithm
        pred_keypoints_2ds, handedness = hand_filtering(pred_keypoints_2ds, handedness, img_size, mp_hand_tracker_bk, debug_vis=True, output_name=debug_image_path)
        # Add to the output
        frame_data["landmarks"] = pred_keypoints_2ds
        frame_data["handedness"] = handedness
    with open(output_path, "w") as f:
        json.dump(raw_data, f)


if __name__ == "__main__":
    # Generate raw data
    raw_process()
