# Gesture-Controlled 3D Canvas

A Python-based 3D drawing application that uses computer vision for hand gesture control. Draw, sculpt, and manipulate 3D shapes in real-time using just your webcam.

## Features

-   **3D Drawing**: Sketch freely in 3D space using your index finger and thumb.
-   **Shape Creation**: Create geometric shapes (Cube, Sphere, Pyramid) by dragging with gestures.
-   **Hand Tracking**: Powered by MediaPipe for robust skeleton detection.
-   **Gesture Controls**: Intuitive pinch and fist gestures for interaction.
-   **Manipulation**: Move and rotate objects or the camera view naturally.
-   **Selection Tools**: Box selection and single-item selection modes.
-   **Undo/Redo**: Fast and responsive history controls.

## Installation

1.  **Clone the repository** (or download usage files).
2.  **Install Dependencies**:
    Ensure you have Python installed. Run the following command to install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

    *Dependencies include: `opencv-python`, `mediapipe`, `pygame`, `PyOpenGL`, `numpy`*

## Usage

Run the main application script:

```bash
python main.py
```

Ensure your webcam is connected and allowed.

## Controls

The application uses specific hand gestures for different actions.

| Action | Hand Gesture | Description |
| :--- | :--- | :--- |
| **Draw / Interact** | **Index + Thumb Pinch** | Pinch your index finger and thumb together to draw strokes or interact with UI sliders. |
| **Menu / UI Select** | **Pinky + Thumb Pinch** | Pinch your pinky and thumb to select tools from the sidebar menu (Undo, Redo, Shapes, Colors). |
| **Rotate / Move** | **Fist (Closed Hand)** | Clench your hand into a fist to grab the scene. Move your hand to rotate the view or move selected objects. |
| **Clear Canvas** | **Open Hand (Spread)** | Spread all fingers open for a moment to clear the entire canvas. A "CLEARING" timer will appear. |
| **Quit** | **Press 'q'** | Press the 'q' key on your keyboard to exit the application. |

## Tips

-   **Lighting**: Ensure your hand is well-lit for best tracking performance.
-   **Mirroring**: The camera feed is mirrored to feel more like a natural mirror.
-   **Selection**: Use the "Select" tool from the menu to grab and move specific objects with the Fist gesture.
