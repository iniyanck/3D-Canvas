import cv2
import time
import numpy as np
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
    
    # Shape Placement State
    shape_anchor_point = None # (x, y, z) when pinch started
    is_placing_shape = False
    was_menu_pinched = False

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
            # --- DEPTH ESTIMATION ---
            # Use hand size to estimate distance from camera (Global Z)
            # Positive = Closer, Negative = Further (roughly -2 to +2 range)
            raw_z = tracker.get_estimated_z(lm_list)
            
            # --- POINTER TRACKING ---
            is_action_pinch, action_pointer = tracker.is_pinching(lm_list, threshold_cm=3.0)
            is_menu_pinch, menu_pointer = tracker.is_menu_pinch(lm_list, threshold_cm=3.0)
            
            # Unpack Pointers
            ax, ay, _ = action_pointer 
            mx, my, _ = menu_pointer
            
            # Smooth the Action Pointer
            # We smooth X, Y, and the Estimated Z (raw_z)
            x, y, z = smoother.update(ax, ay, raw_z)
            
            # Update 3D Cursor position
            canvas.update_cursor(x, y, z)
            
            # --- FIST ROTATION & MOVE (Contextual) ---
            # Priority: Fist blocks everything else
            is_fist = tracker.is_fist_gesture(lm_list)
            
            if is_fist:
                # Wrist tracking for rotation/move
                raw_wx, raw_wy = lm_list[0][1], lm_list[0][2]
                
                # Use filtered Z for the wrist z as well
                wx, wy, wz = wrist_smoother.update(raw_wx, raw_wy, raw_z)
                
                if prev_wrist_x is not None:
                    dx = wx - prev_wrist_x
                    dy = wy - prev_wrist_y
                    dz = wz - prev_wrist_z
                    
                    if canvas.selected_indices:
                        if ui.manipulation_mode == "MOVE":
                            # Move Object
                            canvas.move_selection(dx, dy, dz)
                        else:
                            # Rotate Object
                            canvas.rotate_selection(dx * 0.5, dy * 0.5)
                    else:
                        if ui.manipulation_mode == "MOVE":
                            # Move Camera (Pan/Zoom)
                            canvas.pan_camera(dx, dy, dz)
                        else:
                            # Rotate Camera
                            canvas.rotate(dx * 0.5, dy * 0.5)
                        
                prev_wrist_x, prev_wrist_y, prev_wrist_z = wx, wy, wz
                
                # Exclusivity: Block other actions if Fist is active
                drawing_now = False
                is_action_pinch = False
                is_menu_pinch = False # Also block menu
                
            else:
                prev_wrist_x = None
                wrist_smoother.reset()

            # --- MENU INTERACTION (Pinky + Thumb) ---
            # Only if NO fist
            if not is_fist:
                # Fix Flickering: Only trigger on "Click" (Transition from False -> True)
                if is_menu_pinch and not was_menu_pinched:
                    # Click Event
                    tool_id = ui.check_click(mx, my)
                    
                    if tool_id:
                         # State changes handled in ui.check_click or here
                         if tool_id == "UNDO":
                             canvas.undo()
                             ui.active_tool = "BRUSH" 
                         elif tool_id == "REDO":
                             canvas.redo()
                             ui.active_tool = "BRUSH"
            
                was_menu_pinched = is_menu_pinch
                
                # --- ACTION INTERACTION (Index + Thumb) ---
                if is_action_pinch:
                    active_tool = ui.active_tool
                    
                    if active_tool == "BRUSH":
                        drawing_now = True
                        
                    elif active_tool == "ERASER":
                        # Erase at current pointer
                        canvas.erase_at(x, y, radius=30)
                        cv2.circle(img, (int(x), int(y)), 30, (255, 255, 0), 2)
                        
                    elif active_tool == "SELECT":
                        # Toggle selection
                        if not was_pinched: # Trigger once on start of pinch
                            canvas.select_at(x, y)
                    
                    elif active_tool == "SHAPES":
                        # Drag to Create Logic
                        if not was_pinched:
                            # START Pinch
                            shape_anchor_point = canvas.get_world_point_from_view(x, y, z)
                            is_placing_shape = True
                        
                        # HOLD Pinch (Dragging)
                        if is_placing_shape and shape_anchor_point is not None:
                            current_point = canvas.get_world_point_from_view(x, y, z)
                            canvas.preview_shape_bounds(shape_anchor_point, current_point, ui.active_shape_type)
                
                # Handle Release (Outside is_action_pinch)
                if not is_action_pinch and was_pinched:
                     if is_placing_shape and shape_anchor_point is not None:
                         # RELEASE Pinch
                         current_point = canvas.get_world_point_from_view(x, y, z)
                         canvas.add_shape_bounds(shape_anchor_point, current_point, ui.active_shape_type)
                         is_placing_shape = False
                         shape_anchor_point = None
                         canvas.current_preview_shape = None # Clear preview
                            
                was_pinched = is_action_pinch



            # --- SPREAD CLEAR ---
            if not is_fist and tracker.is_hands_spread_gesture(img):
                if spread_start_time is None: mean_val = 0 # Dummy
                if spread_start_time is None: spread_start_time = time.time()
                if time.time() - spread_start_time > 1.0:
                    canvas.clear()
                    cv2.putText(img, "CLEARED", (width//2 - 100, height//2), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
            else:
                spread_start_time = None
                
            # --- VISUALS ---
            # Draw Action Pointer
            color = (0,255,0) if is_action_pinch else (0,0,255)
            
            # Visualize Depth: Scale Z (-2 to +2 rough) to Radius.
            # User wants: "growing/shrinking dot in the camera as well"
            # We use z (which is filtered estimated_z, +2 close, -2 far).
            # Close (+2) -> Big Radius.
            # Far (-2) -> Small Radius.
            # Map [-2, 2] -> [5, 30]?
            
            # (z + 2) => 0..4
            # * 5 => 0..20
            # + 5 => 5..25
            
            radius = int(5 + (z + 2.0) * 8)
            radius = max(4, min(40, radius))
            
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            
            # Draw Menu Pointer if active
            if is_menu_pinch:
                cv2.circle(img, (int(mx), int(my)), 8, (255,0,255), -1)

        else:
            # No hands
            prev_wrist_x = None
            drawing_now = False

        # 4. Canvas State Update
        start_new = False
        if drawing_now and not was_drawing:
            start_new = True
            
        if drawing_now:
            canvas.add_point(x, y, z, start_new_stroke=start_new)
        elif was_drawing:
            canvas.end_stroke()
            
        was_drawing = drawing_now
        is_drawing = drawing_now

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
