import cv2
import time
from hand_tracking import HandTracker, HandSmoother
from canvas_3d import Canvas3D

def main():
    # Initialize Webcam first to get dimensions
    cap = cv2.VideoCapture(0)
    
    # Apply some settings if possible (optional)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize Hand Tracker
    # model_complexity=1 is default in our modified class
    tracker = HandTracker(detection_con=0.7, track_con=0.7)
    
    # Initialize Smoother
    # alpha=0.5 means 50% new, 50% old. Lower alpha = smoother but more lag.
    # 0.2 is very smooth, 0.7 is more responsive. Let's try 0.3
    smoother = HandSmoother(alpha=0.3)
    
    # Initialize 3D Canvas with camera dimensions
    canvas = Canvas3D(width=width, height=height)
    
    # State
    is_drawing = False
    was_drawing = False

    while True:
        # 1. Capture Frame
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        img = cv2.flip(img, 1) # Mirror image for easier interaction

        # 2. Find Hands (Draws custom skeleton)
        img = tracker.find_hands(img, draw=True)
        lm_list = tracker.find_position(img, draw=False)

        # 3. Gesture Recognition & Smoothing
        x, y, z = 0, 0, 0
        drawing_now = False

        if len(lm_list) != 0:
            # We use Index Finger Tip (8) as the primary pointer
            raw_x, raw_y = lm_list[8][1], lm_list[8][2]
            raw_z = lm_list[8][3]
            
            # Smooth the coordinates
            x, y, z = smoother.update(raw_x, raw_y, raw_z)
            
            # Check for pinch gesture to enable drawing
            # We ignore the returned center from is_drawing_gesture and use our smoothed Index Tip
            # Reduced threshold for finer control. 1.8 cm requires fingers effectively touching.
            pinched, _ = tracker.is_drawing_gesture(lm_list, threshold_cm=1.8)
            
            # Check for eraser gesture (Pinky + Thumb)
            erasing, _ = tracker.is_erasing_gesture(lm_list, threshold_cm=3.5)
            
            # Check for spread hands gesture (Clear Canvas)
            if tracker.is_hands_spread_gesture(img):
                canvas.clear()
                cv2.putText(img, "CLEARED", (300, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
                # Small delay to prevent flickering or multiple clears
                # time.sleep(0.1) # Optional, but might freeze video
            
            if pinched and not erasing:
                drawing_now = True
            
            # Visual feedback for the pointer/cursor
            if erasing:
                color = (255, 255, 0) # Cyan/Yellowish for Eraser
                cv2.circle(img, (x, y), 15, color, cv2.FILLED)
                cv2.putText(img, "ERASING", (x + 20, y - 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                # Perform Erase
                canvas.erase_at(x, y, z, radius=0.15) # Radius is in world coordinates
                
            else:
                color = (0, 255, 0) if drawing_now else (0, 0, 255)
                # Draw a circle at the smoothed position to show where we are really pointing
                cv2.circle(img, (x, y), 10, color, cv2.FILLED)
                if drawing_now:
                     cv2.putText(img, "DRAWING", (x + 15, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            
            # Display X, Y, Z coordinates
            coord_text = f"X: {x} Y: {y} Z: {z:.4f}"
            cv2.putText(img, coord_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
        else:
            # excessive logic to reset smoother could go here if hand is lost
            # smoother.reset() # Optional: if we want to snap when hand reappears
            pass
        
        # 4. Update Canvas
        start_new_stroke = False
        if drawing_now and not was_drawing:
            start_new_stroke = True
        
        if drawing_now:
            canvas.add_point(x, y, z, start_new_stroke=start_new_stroke)
        
        is_drawing = drawing_now
        was_drawing = is_drawing

        # 5. Render 3D Scene
        if not canvas.handle_input():
            break
        canvas.render()

        # 6. Show 2D OpenCV Debug View
        cv2.imshow("Hand Tracking Debug", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
