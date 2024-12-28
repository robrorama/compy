import os
import csv
from PIL import Image
from datetime import datetime

# Paths
SCREENSHOTS_DIR = "screen_shots"
GENERATED_IMAGES_DIR = "generated_images"
TODAY = datetime.now().strftime("%Y-%m-%d")

# Ensure directories exist
os.makedirs(os.path.join(GENERATED_IMAGES_DIR, TODAY), exist_ok=True)

# Read symbols from CSV
def read_symbols(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        symbols = [row[0] for row in reader]
    return symbols

# Crop images
def crop_image(image_path):
    image = Image.open(image_path)
    original_size = image.size
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"Original size: {original_size}")

    # Calculate the cropping area to keep the bottom 60%
    left = 0
    #top = int(original_size[1] * 0.40)  # Keeping the bottom 60% ( original resolution ) 
    top = int(original_size[1] * 0.20)  # Keeping the bottom 60%
    right = original_size[0]
    bottom = original_size[1]

    cropped_image = image.crop((left, top, right, bottom))
    cropped_size = cropped_image.size
    print(f"Cropped size: {cropped_size}")

    # Save the cropped image to the generated_images folder
    cropped_image.save(os.path.join(GENERATED_IMAGES_DIR, TODAY, os.path.basename(image_path)))

if __name__ == "__main__":
    symbols = read_symbols('symbols.csv')
    log_file = open("missing_symbols.log", "w")

    for symbol in symbols:
        screenshot_path = os.path.join(SCREENSHOTS_DIR, TODAY, f"{symbol}.png")
        if os.path.exists(screenshot_path):
            crop_image(screenshot_path)
        else:
            log_file.write(f"{symbol} screenshot is missing\n")
            print(f"{symbol} screenshot is missing")

    log_file.close()

