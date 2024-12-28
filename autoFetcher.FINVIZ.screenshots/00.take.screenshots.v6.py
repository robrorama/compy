import os
import subprocess
import time
import csv
import sys
from datetime import datetime

# Paths
SCREENSHOTS_DIR = "screen_shots"
TODAY = datetime.now().strftime("%Y-%m-%d")

# Ensure directories exist
os.makedirs(os.path.join(SCREENSHOTS_DIR, TODAY), exist_ok=True)

# Check if wmctrl is installed
def check_wmctrl():
    try:
        subprocess.run(["wmctrl", "-h"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: 'wmctrl' is not installed or not found in system path.")
        print("Please install 'wmctrl' using your package manager.")
        exit(1)

# Read symbols from CSV
def read_symbols(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        symbols = [row[0] for row in reader]
    return symbols

# Open Falkon for each URL and take screenshots
def take_screenshots(symbols, period):
    first_time = True

    for symbol in symbols:
        screenshot_path = os.path.join(SCREENSHOTS_DIR, TODAY, f"{symbol}.png")
        if os.path.exists(screenshot_path):
            print(f"Screenshot for {symbol} already exists, skipping download.")
            continue

        print(f"Processing symbol: {symbol}")
        try:
            url = f"https://finviz.com/quote.ashx?t={symbol}&p={period}"
            print(f"Opening Falkon with URL: {url}")
            falkon_process = subprocess.Popen(["falkon", url])
            time.sleep(5)  # Wait for Falkon to open and load the page

            # Locate the Falkon window and bring it to the foreground
            subprocess.run(["wmctrl", "-a", "Falkon"])
            time.sleep(1)  # Give it a moment to focus the window

            if first_time:
                # Make the window full screen only the first time
                subprocess.run(["xdotool", "key", "F11"])
                time.sleep(1)  # Give it a moment to resize the window
                first_time = False

            # Resize the text three times with Ctrl+Shift+Plus

            #for _ in range(3): # OLD RESOLUTION
            for _ in range(1):
                subprocess.run(["xdotool", "key", "ctrl+shift+minus"])
            #    time.sleep(0.5)  # Give it a moment to resize the text

            # Take screenshot
            subprocess.run(["scrot", screenshot_path])  # Assuming 'scrot' is used for screenshots
            print(f"Screenshot saved to {screenshot_path}")

            # Select the Falkon window and close it
            subprocess.run(["wmctrl", "-a", "Falkon"])
            subprocess.run(["xdotool", "key", "ctrl+w"])
            time.sleep(1)  # Ensure the window is closed before proceeding

        except subprocess.CalledProcessError as e:
            print(f"Error processing {symbol}: {e}")
        subprocess.run(["xdotool", "key", "ctrl+w"])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python take_screenshots.py <symbols.csv> <period>")
        print("Period should be one of 'd', 'w', or 'm'")
        exit(1)

    symbols_file = sys.argv[1]
    period = sys.argv[2]

    if period not in ['d', 'w', 'm']:
        print("Period should be one of 'd', 'w', or 'm'")
        exit(1)

    check_wmctrl()
    symbols = read_symbols(symbols_file)
    take_screenshots(symbols, period)

