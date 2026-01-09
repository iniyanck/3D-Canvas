import mediapipe
print("MediaPipe imported successfully")
print(f"Location: {mediapipe.__file__}")
try:
    print(dir(mediapipe))
    print(f"Solutions: {mediapipe.solutions}")
except AttributeError as e:
    print(f"Error accessing solutions: {e}")
