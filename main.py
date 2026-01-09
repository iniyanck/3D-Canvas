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
    
    # Secondary smoother for graceful rotation/zoom (Heavy smoothing)
    wrist_smoother = HandSmoother(alpha=0.1)
    
    # Initialize 3D Canvas with camera dimensions
    canvas = Canvas3D(width=width, height=height)
    
    # State
    is_drawing = False
    was_drawing = False
    prev_wrist_x = None
    prev_wrist_y = None
    prev_wrist_z = None # For Zoom
    spread_start_time = None

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
            
            # Estiamte Z Depth from hand size
            raw_z = tracker.get_estimated_z(lm_list)
            
            # Smooth the coordinates (returns floats now)
            x, y, z = smoother.update(raw_x, raw_y, raw_z)
            
            # Update 3D Cursor position (always)
            canvas.update_cursor(x, y, z)
            
            # Check Fist Gesture for Rotation
            is_fist = tracker.is_fist_gesture(lm_list)
            
            # Check for pinch gesture to enable drawing
            # We ignore the returned center from is_drawing_gesture and use our smoothed Index Tip
            # Increased threshold for easier control. 2.5 cm.
            pinched, _ = tracker.is_drawing_gesture(lm_list, threshold_cm=2.5)
            
            # Check for eraser gesture (Pinky + Thumb)
            erasing, _ = tracker.is_erasing_gesture(lm_list, threshold_cm=3.5)
            
            if is_fist:
                # Disable erasing if rotating
                erasing = False
                
                # Use Wrist (0) for stable tracking
                raw_wx, raw_wy = lm_list[0][1], lm_list[0][2]
                
                # Heavy smoothing for gesture control
                # We use raw_z from above (estimated z)
                wx, wy, wz = wrist_smoother.update(raw_wx, raw_wy, raw_z)
                
                if prev_wrist_x is not None:
                    dx = wx - prev_wrist_x
                    dy = wy - prev_wrist_y
                    
                    # Sensitivity factor for Rotation
                    SENSITIVITY = 0.5
                    canvas.rotate(dx * SENSITIVITY, dy * SENSITIVITY)
                    
                    # Zoom Logic on Z-axis (Forward/Backward)
                    if prev_wrist_z is not None:
                        dz = wz - prev_wrist_z
                        # canvas.zoom(d_z): positive = zoom in.
                        # canvas.zoom(dz)
                        pass
                
                prev_wrist_x, prev_wrist_y = wx, wy
                prev_wrist_z = wz
                # Use raw_wx, raw_wy for text to avoid lag
                cv2.putText(img, "ROTATING", (int(raw_wx), int(raw_wy) - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
            else:
                prev_wrist_x, prev_wrist_y = None, None
                prev_wrist_z = None
                wrist_smoother.reset() # Reset so it doesn't drag from last position
            
            # Check for spread hands gesture (Clear Canvas)
            if tracker.is_hands_spread_gesture(img):
                if spread_start_time is None:
                    spread_start_time = time.time()
                
                elapsed = time.time() - spread_start_time
                if elapsed >= 1.0:
                    canvas.clear()
                    cv2.putText(img, "CLEARED", (300, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
                else:
                    cv2.putText(img, f"HOLD TO CLEAR {1.0-elapsed:.1f}", (300, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 165, 255), 3)
            else:
                spread_start_time = None
            
            if pinched and not erasing and not is_fist:
                drawing_now = True
            
            # Visual feedback for the pointer/cursor
            if erasing:
                # Use midpoint of Pinky(20) and Thumb(4) for erasing
                tx, ty = lm_list[4][1], lm_list[4][2] # Thumb
                px, py = lm_list[20][1], lm_list[20][2] # Pinky
                
                ex, ey = (tx + px) // 2, (ty + py) // 2
                
                color = (255, 255, 0) # Cyan/Yellowish for Eraser
                cv2.circle(img, (int(ex), int(ey)), 15, color, cv2.FILLED)
                cv2.putText(img, "ERASING", (int(ex) + 20, int(ey) - 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                
                # Perform Erase at Eraser Coordinates
                ez = (tracker.get_estimated_z(lm_list)) # Use hand z
                canvas.erase_at(ex, ey, ez, radius=0.15) 
                
            elif is_fist:
                pass # Already handled rotation feedback
                
            else:
                color = (0, 255, 0) if drawing_now else (0, 0, 255)
                # Draw a circle at the smoothed position to show where we are really pointing
                cv2.circle(img, (int(x), int(y)), 10, color, cv2.FILLED)
                if drawing_now:
                     cv2.putText(img, "DRAWING", (int(x) + 15, int(y) - 15), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            
            # Display X, Y, Z coordinates
            coord_text = f"X: {x:.1f} Y: {y:.1f} Z: {z:.4f}"
            cv2.putText(img, coord_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
        else:
            prev_wrist_x, prev_wrist_y = None, None
            prev_wrist_z = None
            spread_start_time = None
            wrist_smoother.reset()
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
