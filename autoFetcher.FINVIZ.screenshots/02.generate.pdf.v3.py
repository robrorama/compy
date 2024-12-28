import os
from PIL import Image
from datetime import datetime

# Paths
GENERATED_IMAGES_DIR = "generated_images"
TODAY = datetime.now().strftime("%Y-%m-%d")

# Generate PDF from cropped images
def generate_pdf():
    image_dir = os.path.join(GENERATED_IMAGES_DIR, TODAY)
    # Get a list of all image filenames, sorted alphabetically
    image_files = sorted([img for img in os.listdir(image_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))])
    # Open images in sorted order
    cropped_images = [Image.open(os.path.join(image_dir, img)) for img in image_files]
    
    if cropped_images:
        # Save the first image and append the rest
        cropped_images[0].save(os.path.join(GENERATED_IMAGES_DIR, f"{TODAY}.pdf"),
                               save_all=True, append_images=cropped_images[1:])
        print(f"PDF generated at {os.path.join(GENERATED_IMAGES_DIR, f'{TODAY}.pdf')}")

if __name__ == "__main__":
    generate_pdf()

