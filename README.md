# ✨ Real-Time 3D Neon Pose Tracker (MediaPipe & OpenCV)

A Python application that uses **MediaPipe Pose Estimation** to detect human body landmarks in real-time, visualizing the pose with a dynamic, pulsing neon effect on a 2D webcam feed and simultaneously rendering a 3D skeletal clone.

## 🚀 Features

* **Real-Time Pose Estimation:** Tracks 33 body landmarks in live video using the fast MediaPipe Pose model.
* **Dynamic Neon Effect:** Overlays the skeletal structure with a glowing, smooth line effect on the webcam feed.
* **Color Cycling & Pulsing:** The neon colors dynamically change over time (hue shift) and pulse in brightness (sine wave intensity).
* **Side-Specific Coloring:** Left and right sides of the body are colored differently for clear visualization.
* **Movement Trail:** Displays a short, fading trail of recent joint positions to visualize movement flow.
* **Interactive 3D Clone:** Renders a synchronized 3D representation of the pose using Matplotlib, updated in real-time.
* **Performance Monitoring:** Displays the real-time FPS (Frames Per Second).

---

## 🛠️ Prerequisites

* Python 3.7+
* **OpenCV** (`cv2`)
* **MediaPipe** (`mediapipe`)
* **Matplotlib** (`matplotlib`)
* **NumPy** (`numpy`)

## 📦 Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies:**
    ```bash
    pip install opencv-python mediapipe matplotlib numpy
    ```

## ⚙️ How to Run

1.  Ensure your webcam is connected and accessible.
2.  Run the main Python script from your terminal:
    ```bash
    python main_tracker.py 
    ```
    *(Note: Replace `main_tracker.py` with the actual name of your script.)*

Press **`Q`** to exit the application.

---

## 💻 Core Code Components

### 1. Model Initialization and Setup

The MediaPipe Pose model is initialized for optimal **video speed** (`model_complexity=0`, `static_image_mode=False`).

