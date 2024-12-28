#!/bin/bash

# Directory containing your images
IMAGE_DIR="./generated_images/"

# List of image files
IMAGE_FILES=$(ls "$IMAGE_DIR")

# Iterate through each image file
for img in $IMAGE_FILES; do
    echo "Opening image: $img"
    
    # Display the image using eog for 1 second
    eog "$IMAGE_DIR/$img" &
    sleep 1
    
    # Find the eog window ID
    WINDOW_ID=$(xdotool search --name "Image Viewer")
    
    # Close the eog window
    xdotool windowkill "$WINDOW_ID"
done

