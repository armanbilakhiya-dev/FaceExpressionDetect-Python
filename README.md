# FaceExpressionDetect-Python

A real-time computer vision project that detects facial expressions and overlays matching stickers using OpenCV and MediaPipe.

## Overview

This project uses facial landmark detection to identify expressions like smile, wink, surprise, and more. Based on the detected expression, a corresponding sticker is displayed next to the face in real time.

It also includes basic hand gesture recognition for detecting a thumbs-up.

## Features

* Real-time face tracking using MediaPipe Face Mesh
* Expression detection based on facial landmarks
* Hand gesture detection (thumbs up)
* Smooth transitions between expressions to avoid flickering
* Transparent PNG sticker overlay
* Works with webcam input

## Supported Expressions

| Expression | Trigger                        |
| ---------- | ------------------------------ |
| Happy      | Smile                          |
| Neutral    | Relaxed face                   |
| Surprise   | Mouth wide open                |
| Angry      | Eyebrows lowered + tight mouth |
| Tease      | Tongue out                     |
| Wink       | One eye closed                 |
| Thumbs     | Thumbs up gesture              |

## Installation

1. Clone the repository

```
git clone https://github.com/your-username/face-expression-sticker.git
cd face-expression-sticker
```

2. Install dependencies

```
pip install -r requirements.txt
```

## Usage

1. Update the emoji folder path inside `main.py`

```
EMOJI_FOLDER = "path_to_emojis_folder"
```

2. Run the project

```
python main.py
```

3. Press `Q` to exit the application.

## Notes

* Make sure your webcam is not being used by another application
* Good lighting improves detection accuracy
* Face should be clearly visible to the camera

## Possible Improvements

* Add more expressions
* Improve detection accuracy with ML model
* Add GUI controls
* Record video output
* Add sound effects

## License

This project is open-source and free to use for learning purposes.
