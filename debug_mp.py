import mediapipe as mp
print("MediaPipe dir:", dir(mp))
try:
    print("mp.solutions:", mp.solutions)
    print("mp.solutions.hands:", mp.solutions.hands)
except Exception as e:
    print("Error accessing solutions:", e)

try:
    import mediapipe.solutions.hands
    print("Direct import success")
except Exception as e:
    print("Direct import error:", e)
