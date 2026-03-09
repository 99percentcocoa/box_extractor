from pathlib import Path

from models import InputImageMeta, DetectionResult
from image_service import detect_apriltags
from extractor import get_row_detections, get_corner_detections, crop_image

if __name__ == "__main__":
    
    input_image_folder = Path("pencil_images")
    files = [f for f in input_image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg"]]

    print(len(files), "files found")

    for file in files:
        print("Processing file:", file)
        image_meta = InputImageMeta(image_path=file)

        # crop image and save
        cropped_image = crop_image(image_meta, Path("pencil_cropped"))
        cropped_image.save(Path("pencil_cropped") / file.name)