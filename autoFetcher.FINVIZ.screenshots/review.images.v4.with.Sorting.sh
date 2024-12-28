#!/bin/bash

# Directory containing your images
IMAGE_DIR="generated_images"

# List of image files
IMAGE_FILES=$(ls "$IMAGE_DIR")

# Function to create directories if they don't exist
create_directories() {
    for dir in "RESULTS" "RESULTS/UPTREND" "RESULTS/DOWNTREND" "RESULTS/SIDEWAYS" "RESULTS/NOTSURE" "RESULTS/XREGENERATE" "RESULTS/FALLINGUPTREND" "RESULTS/RISINGDOWNTREND"; do
        mkdir -p "$dir"
    done
}

# Function to move image based on user input
move_image() {
    read -p "Enter u for UPTREND, d for DOWNTREND, s for SIDEWAYS, n for NOTSURE, x for XREGENERATE, f for FALLINGUPTREND, r for RISINGDOWNTREND: " choice
    case "$choice" in
        u)
            mv "$1" ./RESULTS/UPTREND/
            ;;
        d)
            mv "$1" ./RESULTS/DOWNTREND/
            ;;
        s)
            mv "$1" ./RESULTS/SIDEWAYS/
            ;;
        n)
            mv "$1" ./RESULTS/NOTSURE/
            ;;
        x)
            mv "$1" ./RESULTS/XREGENERATE/
            ;;
        f)
            mv "$1" ./RESULTS/FALLINGUPTREND/
            ;;
        r)
            mv "$1" ./RESULTS/RISINGDOWNTREND/
            ;;
        *)
            echo "Invalid choice. Skipping file."
            ;;
    esac
}

# Create directories if they don't exist
create_directories

# Iterate through each image file
for img in $IMAGE_FILES; do
    echo "Opening image: $img"
    
    # Display the image using eog and wait for it to be closed manually
    eog "$IMAGE_DIR/$img"
    
    # Prompt user for destination directory and move the image
    move_image "$IMAGE_DIR/$img"
done

