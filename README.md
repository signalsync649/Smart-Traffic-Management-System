Smart-Traffic-Management-System

AI-based traffic management system that optimizes signal timing using real-time vehicle detection.

Getting Started

Prerequisites:

Python 3.8 or higher

Node.js and npm installed

Optional: GPU with CUDA support for faster processing

Installation Steps:

Clone this repository by running the following commands:

git clone <repository-url>
cd <repository-folder>

Install Python dependencies. Open the terminal in VS Code and run:
pip install opencv-python numpy torch ultralytics requests pytz

Install Node.js packages by running:

npm install

Install system dependencies (if required) for video file support (ffmpeg):

On Ubuntu/Debian:

sudo apt update
sudo apt install ffmpeg

On macOS (with Homebrew):

brew install ffmpeg

On Windows:

Download ffmpeg from https://ffmpeg.org/download.html
 and add it to your system PATH.

Running the Project:

Start the Python API server by running:

python api_server.py

Start the Node.js development server by running:

npm run dev

Start the ML script by running:

python ml.py

Verify Installation:

You can check that all required Python packages are installed by running:
python -c "import cv2; import numpy; import torch; import requests; import pytz; print('All dependencies are installed!')"

Now you're ready to explore and run the Smart Traffic Management System!
