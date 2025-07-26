# Car Damage Detection Using YOLOv8

This repository contains a Flask web application demonstrating how to detect vehicle damage using the **YOLOv8** object detection model. Users can upload images or videos or stream from a webcam to locate damaged regions on cars.

## Features

- **Image Detection** – upload a picture of a vehicle and visualize detected damage.
- **Video Processing** – submit a video file to produce an annotated video showing each detected area.
- **Live Webcam** – perform real‑time detection straight from your webcam.
- **Pretrained Model** – includes the `best.pt` weights trained for car damage detection.

A short example run is available in `execution video.mp4`.

## Project Structure

```
BASE PAPER/                          # Reference paper
SOURCE CODE/
└─ car damage detection using yolov8/
   ├─ app.py                         # Flask application
   ├─ best.pt                        # YOLOv8 weights
   ├─ templates/                     # HTML templates
   ├─ static/                        # CSS/JS/assets
   └─ uploads/                       # Sample media files
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-user>/Car-Damage-Detection.git
   cd Car-Damage-Detection/Enhancing\ Car\ Damage\ Analysis\ through\ YOLOV8\ model/SOURCE\ CODE/car\ damage\ detection\ using\ yolov8
   ```

2. **Install dependencies** (preferably inside a virtual environment)
   ```bash
   pip install flask opencv-python pillow ultralytics
   ```

3. **Run the app**
   ```bash
   python app.py --port 5000
   ```
   Open `http://localhost:5000` in a browser and choose between image, video, or webcam detection.

## Dataset and Model

The provided `best.pt` file was trained on a curated dataset of damaged vehicles. Media files placed in the `uploads/` directory are used to try the application.

## Reference

See `BASE PAPER/A System for Automated Vehicle Damage.pdf` for more background on automated damage assessment.

## License

This project is intended for educational use. Please adapt the license to your requirements.

