# ASL-to-English Interpreter

A real-time **American Sign Language (ASL) recognition system** that translates hand gestures into English text using **OpenVINO**, **OpenCV**, and **Python**.
This project demonstrates efficient video inference pipelines for sign language interpretation, optimized for low latency and high accuracy.

---

## Features

* **Real-Time Gesture Recognition** – Detects and classifies ASL gestures from live camera input.
* **Frame Preprocessing & Motion Detection** – Improves recognition stability and responsiveness.
* **Multi-Threaded Pipeline** – Runs video capture and model inference in parallel for efficiency.
* **Automatic Transcript Saving** – Saves recognized text with timestamps for later review.
* **Configurable Parameters** – Adjust model path, device type, thresholds, and logging via CLI.

---

## 🛠️ Technologies Used

* **Python 3.8+**
* **OpenVINO Toolkit** – for model inference acceleration
* **OpenCV** – for video capture and visualization
* **NumPy** – for array operations and preprocessing

---

## ▶️ Usage

Run the application with default settings:

```bash
python main.py
```

Or specify custom parameters:

```bash
python main.py --model model/asl.xml --device CPU --threshold 0.7 --smooth 5
```

| Argument      | Description                                   | Default         |
| ------------- | --------------------------------------------- | --------------- |
| `--model`     | Path to OpenVINO model file (.xml)            | `model/asl.xml` |
| `--device`    | Inference device (`CPU`, `GPU`, `AUTO`, etc.) | `AUTO`          |
| `--threshold` | Minimum confidence for prediction             | `0.6`           |
| `--smooth`    | Number of frames for temporal smoothing       | `5`             |

---




