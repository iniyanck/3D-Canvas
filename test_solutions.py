import mediapipe as mp
print(f"Version: {mp.__version__}")

try:
    print(f"Solutions: {mp.solutions}")
except AttributeError:
    print("Direct access to mp.solutions failed")

try:
    import mediapipe.python.solutions as solutions
    print("Import mediapipe.python.solutions succeeded")
    print(f"Solutions module: {solutions}")
except ImportError as e:
    print(f"Import mediapipe.python.solutions failed: {e}")
except AttributeError as e:
    print(f"Attribute error during import: {e}")
