#!/bin/bash

# Directory containing your images
#IMAGE_DIR="/path/to/your/images"
IMAGE_DIR="./generated_images/"



# List of image files
IMAGE_FILES=$(ls "$IMAGE_DIR")

# Iterate through each image file
for img in $IMAGE_FILES; do
    # Display the image using eog
    eog "$IMAGE_DIR/$img" &
    sleep 1  # Adjust sleep time if necessary to ensure window is fully loaded

    # Find the eog window ID
    WINDOW_ID=$(xdotool search --name "Image Viewer")
    
    # Make the window full-screen
    xdotool windowactivate "$WINDOW_ID"
    xdotool key F11  # Press F11 to toggle full-screen (assuming F11 toggles full-screen in eog)
    
    # Wait for any keystroke
    read -n 1 -s -r -p "Press any key to continue to the next image..."
    
    # Close the eog window
    xdotool windowkill "$WINDOW_ID"
done

