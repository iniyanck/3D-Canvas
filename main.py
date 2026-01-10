import cv2
import time
from hand_tracking import HandTracker, HandSmoother
from canvas_3d import Canvas3D
from ui_overlay import UIOverlay

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
    ui = UIOverlay(width=width, height=height)
    
    # State
    is_drawing = False
    was_drawing = False
    was_pinched = False # To detect click vs hold
    
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
            
            # Estimate Z Depth from hand size
            raw_z = tracker.get_estimated_z(lm_list)
            
            # Smooth the coordinates (returns floats now)
            x, y, z = smoother.update(raw_x, raw_y, raw_z)
            
            # Update 3D Cursor position (always)
            canvas.update_cursor(x, y, z)
            
            # Get 2D Interface Position for UI interaction
            cursor_pos_2d = canvas.get_interface_position(x, y, z)
            if cursor_pos_2d:
                cx, cy = cursor_pos_2d
            else:
                cx, cy = raw_x, raw_y # Fallback if behind camera?
            
            # Check Fist Gesture for Rotation/Manipulation
            is_fist = tracker.is_fist_gesture(lm_list)
            
            # Check for pinch gesture (Action Trigger)
            # Increased threshold for easier control. 2.5 cm.
            pinched, _ = tracker.is_drawing_gesture(lm_list, threshold_cm=2.5)
            pinch_start = pinched and not was_pinched
            
            # --- FIST LOGIC (Rotation / Manipulation) ---
            if is_fist:
                # Use Wrist (0) for stable tracking
                raw_wx, raw_wy = lm_list[0][1], lm_list[0][2]
                
                # Heavy smoothing for gesture control
                # We use raw_z from above (estimated z)
                wx, wy, wz = wrist_smoother.update(raw_wx, raw_wy, raw_z)
                
                if prev_wrist_x is not None:
                    dx = wx - prev_wrist_x
                    dy = wy - prev_wrist_y
                    
                    SENSITIVITY = 0.5
                    
                    # If selection exists and Select tool active (or maybe always if selection?)
                    # Let's say if Selection Exists, Fist manipulates selection, unless we are in Camera Mode?
                    # User said: "Or if you just do the rotate/pan while selected, it moves around/rotates the selected shape"
                    if canvas.selected_indices:
                        # Move Selection
                        canvas.move_selection(dx, dy)
                        # Rotate Selection? Maybe separate gesture or mode?
                        # Let's just do Move for now with Fist Pan.
                        cv2.putText(img, "MOVING SELECTION", (int(raw_wx), int(raw_wy) - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                    else:
                        # Rotate Camera
                        canvas.rotate(dx * SENSITIVITY, dy * SENSITIVITY)
                        cv2.putText(img, "ROTATING VIEW", (int(raw_wx), int(raw_wy) - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
                    
                    # Zoom Logic on Z-axis (Forward/Backward)
                    if prev_wrist_z is not None:
                        dz = wz - prev_wrist_z
                        # canvas.zoom(dz)
                        pass
                
                prev_wrist_x, prev_wrist_y = wx, wy
                prev_wrist_z = wz
            else:
                prev_wrist_x, prev_wrist_y = None, None
                prev_wrist_z = None
                wrist_smoother.reset() # Reset so it doesn't drag from last position

            # --- PINCH LOGIC (UI & Tools) ---
            if pinched:
                # 1. UI Interaction
                if cx < ui.panel_width:
                    if pinch_start: # Click only once
                        action = ui.check_click(cx, cy)
                        if action:
                            if action == "UNDO":
                                canvas.undo()
                            elif action == "REDO":
                                canvas.redo()
                            # Tool switching is handled inside check_click updating active_tool
                else:
                    # 2. Canvas Interaction
                    tool = ui.active_tool
                    
                    if tool == "BRUSH":
                         # Draw
                         drawing_now = True
                         
                    elif tool == "ERASER":
                        # Erase continuously while pinched
                         canvas.erase_at(cx, cy, radius=30)
                         cv2.circle(img, (int(cx), int(cy)), 30, (255, 255, 0), 2)
                         
                    elif tool == "SELECT":
                         if pinch_start:
                             canvas.select_at(cx, cy)
                             
                    elif tool == "SHAPES":
                         if pinch_start:
                             # Add cube for now
                             canvas.add_shape("CUBE", x, y, z)
                             # Switch back to Select or Brush? Or keep adding shapes?
                             # Keep adding.
            
            was_pinched = pinched
            
            # --- Visual Feedback ---
            color = (0, 255, 0) if pinched else (0, 0, 255)
            # Show cursor 2D position
            if cursor_pos_2d:
                 cv2.circle(img, (int(cx), int(cy)), 10, color, cv2.FILLED)
                 if drawing_now:
                      cv2.putText(img, "DRAWING", (int(cx) + 15, int(cy) - 15), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            
            # Display Coordinates
            coord_text = f"X: {x:.1f} Y: {y:.1f} Z: {z:.4f}"
            cv2.putText(img, coord_text, (100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            
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
            
        else:
            prev_wrist_x, prev_wrist_y = None, None
            wrist_smoother.reset()
            was_pinched = False
        
        # 4. Update Canvas (Drawing strokes)
        start_new_stroke = False
        if drawing_now and not was_drawing:
            start_new_stroke = True
        
        if drawing_now:
            canvas.add_point(x, y, z, start_new_stroke=start_new_stroke)
        elif was_drawing:
            canvas.end_stroke()
        
        is_drawing = drawing_now
        was_drawing = is_drawing

        # 5. Render 3D Scene
        if not canvas.handle_input():
            break
        canvas.render()
        
        # 6. Render UI Overlay on top of Camera Feed
        ui.draw(img)

        # 7. Show 2D OpenCV Debug View
        cv2.imshow("Hand Tracking Debug", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
