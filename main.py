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
    
    manipulation_end_time = 0
    was_fist = False

    spread_start_time = None
    last_spread_valid_time = 0

    
    # Shape Placement State
    shape_anchor_point = None # (x, y, z) when pinch started
    is_placing_shape = False
    was_menu_pinched = False
    last_menu_click_time = 0
    
    cleared_display_start_time = None

    
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
            
            # --- FIST ROTATION & MOVE (Contextual) ---
            # Priority: Fist blocks everything else
            is_fist = tracker.is_fist_gesture(lm_list)
            
            if is_fist:
                canvas.cursor_pos = None # Hide cursor during manipulation
            else:
                # Update 3D Cursor position only if not manipulating
                canvas.update_cursor(x, y, z)
            
            if is_fist:
                # Wrist tracking for rotation/move
                raw_wx, raw_wy = lm_list[0][1], lm_list[0][2]
                
                # Use filtered Z for the wrist z as well
                wx, wy, wz = wrist_smoother.update(raw_wx, raw_wy, raw_z)
                
                if prev_wrist_x is not None:
                    dx = wx - prev_wrist_x
                    dy = wy - prev_wrist_y
                    dz = wz - prev_wrist_z
                    
                    if canvas.selected_indices or canvas.selected_shape_indices:
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
                
                was_fist = True
                
                # Cancel Shape Placement & Reset States to prevent overlap/glitches
                is_placing_shape = False
                shape_anchor_point = None
                canvas.current_preview_shape = None
                
                # Reset touch history so we don't trigger "click on release" after rotation ends
                was_pinched = False
                was_menu_pinched = False
                
            else:
                if was_fist:
                    manipulation_end_time = time.time()
                    was_fist = False

                prev_wrist_x = None
                wrist_smoother.reset()

            # --- SPREAD CLEAR ---
            is_cleared_message_showing = (cleared_display_start_time is not None) and (time.time() - cleared_display_start_time < 2.0)
            is_spread = (not is_fist) and tracker.is_hands_spread_gesture(img)
            
            # Debounce Logic:
            # If we see the gesture, update last valid time.
            if is_spread:
                last_spread_valid_time = time.time()
            
            # If we recently saw it (within 0.2s), consider it still active.
            # This prevents flickering from resetting the timer.
            is_spread_active = (time.time() - last_spread_valid_time < 0.2)

            # Only allow clearing if we are NOT currently showing the "CLEARED" message
            if is_spread_active and not is_cleared_message_showing:
                if spread_start_time is None: 
                    spread_start_time = time.time()
                
                # Feedback
                elapsed = time.time() - spread_start_time
                required_time = 0.5 # Reduced to 0.5s for speed
                
                if elapsed < required_time:

                    # Show Progress (Remaining Time)
                    remaining = max(0.0, required_time - elapsed)
                    # Countdown: Ceiling to make it 1.0, 0.9... until 0
                    cv2.putText(img, f"CLEARING {remaining:.1f}s", (width//2 - 100, height//2), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                else:
                    canvas.clear()
                    spread_start_time = None # Reset
                    cleared_display_start_time = time.time()
            else:
                spread_start_time = None
            
            if cleared_display_start_time and time.time() - cleared_display_start_time < 2.0:
                 cv2.putText(img, "CLEARED", (width//2 - 100, height//2), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
            elif cleared_display_start_time:
                 cleared_display_start_time = None
                
            # --- MENU INTERACTION (Pinky + Thumb) ---
            # Only if NO fist
            if not is_fist:
                
                # Determine if we should process UI input
                # 1. Clicks (Button taps) -> Trigger on transition to True
                # 2. Dragging (Slider/Picker) -> Trigger continuously while True if in valid area
                
                process_ui = False
                
                if is_menu_pinch:
                     # Check if we are interacting with a slider (Submenu active and pointer in submenu region)
                     if ui.active_submenu:
                         sub_x = ui.x + ui.w + 20 # Approximate start of submenu
                         if mx > sub_x:
                             process_ui = True # Allow continuous
                     
                     # Check for standard click (transition)
                     if not was_menu_pinched:
                         # Debounce: prevent spam clicks (flicker)
                         if time.time() - last_menu_click_time > 0.35:
                             process_ui = True
                             last_menu_click_time = time.time()
                
                if process_ui:
                     prev_tool = ui.active_tool
                     tool_id = ui.check_click(mx, my)
                     
                     if tool_id:
                         # Clear selection if tool changed
                         if ui.active_tool != prev_tool:
                             canvas.clear_selection()

                         # Handle "Global" commands that affect canvas state directly
                         if tool_id == "UNDO":
                             canvas.undo()
                         elif tool_id == "REDO":
                             canvas.redo()
                         # UPDATE_SETTINGS and others are handled inside UI state
            
                was_menu_pinched = is_menu_pinch
                
                # --- ACTION INTERACTION (Index + Thumb) ---
                # Add Cooldown check: Don't allow pinch immediately after manipulation (0.5s)
                if is_action_pinch and (time.time() - manipulation_end_time > 0.5):
                    active_tool = ui.active_tool
                    
                    # Sync Settings to Canvas
                    if "BRUSH" in active_tool or "ERASER" in active_tool:
                        canvas.set_color(ui.brush_color)
                        canvas.set_thickness(ui.brush_thickness)
                    elif "SHAPES" in active_tool or active_tool.startswith("SHAPE_"):
                        canvas.set_color(ui.shape_color)
                        canvas.current_thickness = 1 # Reset or ignore for shapes
                    
                    if active_tool == "BRUSH":
                        # Only draw if NOTHING is selected. 
                        # If something is selected, user likely wants to manipulate it.
                        if not canvas.selected_indices and not canvas.selected_shape_indices:
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
                            # Standardize to World Alignment for consistent placement
                            # shape_rot = 0.0 # Old
                            shape_rot = -canvas.rot_y # Align with Camera
                            canvas.preview_shape_bounds(shape_anchor_point, current_point, ui.active_shape_type, rotation=shape_rot)
                
                # Handle Release (Outside is_action_pinch)
                if not is_action_pinch and was_pinched:
                     if is_placing_shape and shape_anchor_point is not None:
                         # RELEASE Pinch
                         current_point = canvas.get_world_point_from_view(x, y, z)
                         shape_rot = -canvas.rot_y # Align with Camera
                         canvas.add_shape_bounds(shape_anchor_point, current_point, ui.active_shape_type, rotation=shape_rot)
                         is_placing_shape = False
                         shape_anchor_point = None
                         canvas.current_preview_shape = None # Clear preview
                            
                was_pinched = is_action_pinch

            # --- SPREAD CLEAR (Backup/Duplicate check removed to prevent conflicts) ---
            # The logic is already handled in lines 135-150.
            pass
                
            # --- VISUALS ---
            # Draw Action Pointer
            color = (0,255,0) if is_action_pinch else (0,0,255)
            
            # Visualize Depth
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
