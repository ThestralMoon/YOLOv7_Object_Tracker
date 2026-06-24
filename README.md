# YOLOv7 Object Tracker

A Python-based object detection and tracking project that combines **YOLOv7** for real-time object detection with **ByteTrack** for multi-object tracking.

The tracker can process video files or single image frames, draw bounding boxes around detected objects, assign tracking IDs, and count detected people in the frame.

## Demo

[Watch the demo video](https://www.youtube.com/watch?feature=player_embedded&v=64Ys7Xj4Iho)

## Features

- Detects objects using YOLOv7
- Tracks objects across video frames with ByteTrack
- Draws labeled bounding boxes on detected objects
- Assigns tracking IDs to objects in video
- Counts detected people in frames
- Supports both video tracking and single-frame detection
- Runs through Docker for easier setup

## Project Structure

```text
YOLOv7_Object_Tracker/
├── detections.py            # Drawing, formatting, and counting detections
├── obj_detector.py          # YOLOv7 + ByteTrack detector wrapper
├── track_objects.py         # Tracks objects in a video file
├── track_single_frame.py    # Runs detection/tracking on one image frame
├── yolov7/                  # YOLOv7 model code and dependencies
├── resources/               # Additional project resources
└── README.md
```

## Installation

### Run with Docker

Pull the Docker image:

```bash
docker pull thestralmoon/yolov7_object_tracker:first_commit
```

Create and start a container:

```bash
docker run --name yolov7_tracker -t -d thestralmoon/yolov7_object_tracker:first_commit
```

Enter the container:

```bash
docker exec -it yolov7_tracker bash
```

## Usage

### Track Objects in a Video

Run:

```bash
python track_objects.py \
  --device cpu \
  --video VIRAT_S_000001.mp4 \
  --output VIRAT_S_000001_Tracked.mp4
```

Arguments:

| Argument | Description | Example |
|---|---|---|
| `--device` | Device used for inference. Use `cpu` or a CUDA device if available. | `cpu` |
| `--video` | Input video path. | `input.mp4` |
| `--output` | Output video path. | `tracked_output.mp4` |
| `--weights` | Optional path to YOLOv7 weights. | `yolov7/yolov7.pt` |
| `--classes` | Optional path to class configuration YAML. | `track_classes.yaml` |

### Track a Single Frame

Run:

```bash
python track_single_frame.py \
  --device cpu \
  --image_file test_frame.png \
  --output detected_test_frame.png
```

Arguments:

| Argument | Description | Example |
|---|---|---|
| `--device` | Device used for inference. | `cpu` |
| `--image_file` | Input image path. | `test_frame.png` |
| `--output` | Output image path. | `detected_test_frame.png` |
| `--weights` | Optional path to YOLOv7 weights. | `yolov7/yolov7.pt` |
| `--classes` | Optional path to class configuration YAML. | `track_classes.yaml` |

## Output Example

The single-frame script saves an annotated image with bounding boxes, labels, confidence scores, and a people count.

![Detected test frame](https://github.com/ThestralMoon/YOLOv7_Object_Tracker/assets/31570034/1d4179d2-1210-46e0-a1e4-27f4a5569681)

## Class Configuration

The tracker expects a class configuration file, usually named:

```text
track_classes.yaml
```

This file should define the object classes and display colors used when drawing detections.

Example format:

```yaml
classes:
  - name: person
    color: "#FF3838"
  - name: bicycle
    color: "#FF9D97"
  - name: car
    color: "#FF701F"
```

Important: the class list should match the order of the model weights you are using. For COCO-trained YOLOv7 weights, use the COCO class order.

## Notes

- Docker is the recommended setup method for this project.
- CPU inference is supported, but GPU inference is recommended for faster video processing.
- Make sure your input video or image file is available inside the Docker container before running the scripts.
- The default model weights path is `yolov7/yolov7.pt`.

## Technologies Used

- Python
- OpenCV
- PyTorch
- YOLOv7
- ByteTrack
- Docker

## Future Improvements

- Add a complete `track_classes.yaml` file to the repository
- Add source-install instructions without Docker
- Add GPU Docker instructions
- Add more sample input/output files
- Add performance benchmarks such as FPS on CPU and GPU
- Add support for webcam or live camera streams

## Acknowledgments

This project builds on:

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
