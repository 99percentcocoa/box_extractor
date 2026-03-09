from pathlib import Path
import random
from typing import List

from models import InputImageMeta, DetectionResult
from image_service import detect_apriltags

import cv2
import numpy as np

from config import SETTINGS


# no need to preprocess, start from row detections

def get_row_detections(image: InputImageMeta) -> DetectionResult:
    """Get row detections from the input image.
    
    Args:
        image: InputImageMeta object containing the image to process.
    
    Returns:
        DetectionResult containing the input image and detected rows.
    """
    if image.image_array is None:
        raise ValueError("Input image array is None")
    
    detections = detect_apriltags(image, tag_family="25h9")
    return detections

def get_corner_detections(image: InputImageMeta) -> DetectionResult:
    """Get corner detections from the input image.
    
    Args:
        image: InputImageMeta object containing the image to process.
    
    Returns:
        DetectionResult containing the input image and detected rows.
    """
    if image.image_array is None:
        raise ValueError("Input image array is None")
    
    detections = detect_apriltags(image, tag_family="36h11")
    return detections

def get_rois(image: InputImageMeta):

    row_detections = get_row_detections(image)
    roi_images = []

    # loop through row detections and extract ROIs
    for det in row_detections.detections:
        print("In tag number :", det.tag_id)
        center_x, center_y = int(det.center[0]), int(det.center[1])
        
        for roi in (SETTINGS.LEFT_QUESTION_ROI, SETTINGS.RIGHT_QUESTION_ROI):
            x_offset, y_offset, width, height = roi
            
            # Calculate top-left corner of ROI based on detection center
            x1 = center_x + x_offset
            y1 = center_y + y_offset
            x2 = x1 + width
            y2 = y1 + height
            
            # Extract ROI from image
            roi_image = InputImageMeta(image_array=image.image_array[y1:y2, x1:x2])
            
            # Split roi_image into 4 equal parts horizontally
            roi_array = roi_image.image_array
            roi_width = roi_array.shape[1]
            part_width = roi_width // 4
            
            for i in range(4):
                start_x = i * part_width
                end_x = start_x + part_width if i < 3 else roi_width
                roi_part = InputImageMeta(image_array=roi_array[:, start_x:end_x])
                roi_images.append(roi_part)
    
    return roi_images

def save_rois(filename: str, roi_images: List[InputImageMeta], output_folder: Path):
    """Save extracted ROI images to the specified output folder."""
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for idx, roi_image in enumerate(roi_images):
        output_path = output_folder / f"{filename}_roi_{idx+1}.jpg"
        cv2.imwrite(str(output_path), roi_image.image_array)
        print(f"Saved ROI image to: {output_path}")

def crop_image(image: InputImageMeta, output_folder: Path):
    """Crop the input image into ROIs based on detected rows and save them."""
    corner_detections = get_corner_detections(image)
    
    if len(corner_detections.detections) < 4:
        raise ValueError("Not enough corner detections found.")
    
    # Use sorted detections: top-left, top-right, bottom-right, bottom-left
    ordered = corner_detections.sorted_detections
    if len(ordered) < 4:
        raise ValueError("Sorted detections did not yield 4 corners.")

    # Build source points (x, y) in float32 shape (4,2)
    src_pts = np.array([ordered[i].center for i in range(4)], dtype=np.float32)
    dst_pts = np.array([[0, 0], [SETTINGS.TARGET_WIDTH, 0], [SETTINGS.TARGET_WIDTH, SETTINGS.TARGET_HEIGHT], [0, SETTINGS.TARGET_HEIGHT]], dtype="float32")

    # Compute perspective transform matrix
    t_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perform the warp perspective to get the cropped image
    warped_image = cv2.warpPerspective(image.image_array, t_matrix, (SETTINGS.TARGET_WIDTH, SETTINGS.TARGET_HEIGHT))

    return InputImageMeta(image_array=warped_image)

if __name__ == "__main__":
    
    input_image_folder = Path("pencil_cropped")
    files = [f for f in input_image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg"]]

    print(len(files), "files found")

    # random_file = random.choice(files)
    # print("Processing file:", random_file)

    # print(random_file)

    # crop image and save
    # cropped_image = crop_image(image_meta, Path("input_images"))
    # cropped_image.save(Path("input_images") / random_file.name)

    # detect

    log_file = Path("error_log.txt")
    
    # for file in files:
    #     image_meta = InputImageMeta(image_path=file)

    #     try:
    #         roi_images = get_rois(image_meta)
    #     except ValueError as e:
    #         print(f"Error processing {file.name}: {e}")
            
    #         with open(log_file, "a") as f:
    #             f.write(f"{file.name}: {e}\n")

    #         continue
    #     save_rois(file.stem, roi_images, Path("pencil_output"))
    
    # single file
    image_meta = InputImageMeta(image_path='cropped_images/9.jpeg')
    try:
        roi_images = get_rois(image_meta)
    except ValueError as e:
        print(f"Error processing {image_meta.image_path.name}: {e}")
    save_rois(image_meta.image_path.stem, roi_images, Path("pencil_output/9"))