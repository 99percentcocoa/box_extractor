from pathlib import Path

from models import InputImageMeta, DetectionResult
from image_service import detect_apriltags
from extractor import get_row_detections, get_corner_detections

if __name__ == "__main__":
    
    input_image_folder = Path("downloadscopy")
    files = [f for f in input_image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg"]]

    print(len(files), "files found")

    for file in files:
        print("Processing file:", file)
        image_meta = InputImageMeta(image_path=file)

        try:
            corner_detections = get_corner_detections(image_meta)
            row_detections = get_row_detections(image_meta)
            print(f"Detected {len(corner_detections.detections)} corners in image: {file.name}")
        except Exception as e:
            print(f"Error processing image {file.name}: {e}")
            print(f"Deleting file {file}")
            file.unlink()
