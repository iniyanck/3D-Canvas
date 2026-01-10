import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Has solutions: {hasattr(mp, 'solutions')}")
    print(mp.solutions.hands)
except Exception as e:
    print(f"Error: {e}")
