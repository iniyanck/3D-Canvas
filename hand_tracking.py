import mediapipe as mp
import cv2
import numpy as np
import math

class HandSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None

    def update(self, x, y, z):
        if self.prev_x is None:
            self.prev_x, self.prev_y, self.prev_z = x, y, z
            return x, y, z
        
        # Exponential Moving Average
        # New value has weight 'alpha', old value has weight '1-alpha'
        # Or: smooth = prev + alpha * (new - prev) -> smooth = prev * (1-alpha) + new * alpha
        
        smooth_x = self.prev_x * (1 - self.alpha) + x * self.alpha
        smooth_y = self.prev_y * (1 - self.alpha) + y * self.alpha
        smooth_z = self.prev_z * (1 - self.alpha) + z * self.alpha
        
        self.prev_x, self.prev_y, self.prev_z = smooth_x, smooth_y, smooth_z
        return smooth_x, smooth_y, smooth_z

    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
            model_complexity=1 # 1 is good balance, 0 is fast, 2 is accurate
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        
        # Tip IDs for all fingers (Thumb, Index, Middle, Ring, Pinky)
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """
        Processes the image to find hands. 
        If draw=True, draws custom skeletons for Index and Thumb only.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw only Thumb (1-4) and Index (5-8)
                    # We can use mp_draw for specific connections or manual lines
                    self._draw_custom_hand(img, hand_lms)
                    
        return img

    def _draw_custom_hand(self, img, hand_lms):
        h, w, c = img.shape
        
        # Define connections we care about (All fingers)
        # Thumb: 0->1->2->3->4
        # Index: 0->5->6->7->8
        # Middle: 0->9->10->11->12
        # Ring: 0->13->14->15->16
        # Pinky: 0->17->18->19->20
        
        connections = [
            (1, 0), (2, 1), (3, 2), (4, 3),   # Thumb
            (5, 0), (6, 5), (7, 6), (8, 7),   # Index
            (9, 0), (10, 9), (11, 10), (12, 11), # Middle
            (13, 0), (14, 13), (15, 14), (16, 15), # Ring
            (17, 0), (18, 17), (19, 18), (20, 19)  # Pinky
        ]
        
        # Get all landmarks in pixel coords
        lms_px = []
        for lm in hand_lms.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lms_px.append((cx, cy))
            
        # Draw lines only (Skeleton)
        for start_idx, end_idx in connections:
            if start_idx < len(lms_px) and end_idx < len(lms_px):
                cv2.line(img, lms_px[start_idx], lms_px[end_idx], (255, 255, 255), 3)

    def find_position(self, img, hand_no=0, draw=False): # draw here is for legacy circle drawing
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            try:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Return pixel x,y and normalized z
                    lm_list.append([id, cx, cy, lm.z])
            except IndexError:
                pass
        return lm_list

    def get_3d_distance_cm(self, lm_list, p1_idx, p2_idx):
        """
        Calculate the approximate 3D distance between two points in centimeters.
        Uses the hand size (Wrist 0 to Middle MCP 9) as a reference scale of ~10cm.
        """
        if len(lm_list) < 21:
            return float('inf')

        # 1. Calculate Scale Factor (Centimeters per Pixel)
        # Reference: Wrist (0) to Middle MCP (9)
        # We assume an average hand size for this segment is roughly 9-10 cm.
        # Let's align with a standard: ~ 10cm seems reasonable for an adult hand.
        
        # Note: We don't have MCP in our `lm_list` because `find_position` returns a list 
        # but MediaPipe indices are consistent. 
        # But wait, `find_position` iterates `enumerate(my_hand.landmark)`, so `lm_list` has ALL 21 landmarks.
        # However, check if `find_position` output matches standard indices.
        # Yes: lm_list[id] -> [id, cx, cy, z]
        
        x0, y0 = lm_list[0][1], lm_list[0][2]
        x9, y9 = lm_list[9][1], lm_list[9][2]
        
        # 2D pixel distance of reference segment
        ref_pixel_dist = math.hypot(x9 - x0, y9 - y0)
        
        if ref_pixel_dist == 0: 
            return float('inf')
            
        REF_CM = 10.0 # Standard size of Wrist to Middle Knuckle
        cm_per_pixel = REF_CM / ref_pixel_dist
        
        # 2. Calculate Distance between target points
        x1, y1 = lm_list[p1_idx][1], lm_list[p1_idx][2]
        x2, y2 = lm_list[p2_idx][1], lm_list[p2_idx][2]
        
        # Pixel distance
        dist_px = math.hypot(x2 - x1, y2 - y1)
        
        # 3. Convert to CM
        # Note: We are ignoring relative Z depth between fingers for the pinch *distance* for now
        # because for pinching, X/Y distance is dominant and Z is usually similar.
        # If we want true 3D, we need proper Z pixel-equivalent which is harder without intrinsics.
        # But scaling the 2D projected distance by our depth-aware scale factor is exactly what we want!
        
        dist_cm = dist_px * cm_per_pixel
        dist_cm = dist_px * cm_per_pixel
        return dist_cm

    def get_estimated_z(self, lm_list):
        """
        Estimate Z depth based on hand size (Wrist to Middle MCP).
        Returns a Z value where 0 is 'standard' distance, positive is closer, negative is farther.
        """
        if len(lm_list) < 21:
            return 0.0
        
        # Wrist (0) to Middle MCP (9)
        x0, y0 = lm_list[0][1], lm_list[0][2]
        x9, y9 = lm_list[9][1], lm_list[9][2]
        
        # Calculate size in pixels
        current_size = math.hypot(x9 - x0, y9 - y0)
        
        # Standard size calibration (adjust as needed)
        # Assuming ~100-150 pixels is a comfortable distance
        STD_SIZE = 120.0
        
        # Simple linear mapping for Z relative to camera
        # If hand is double size (240), it's much closer.
        # We scale this to be suitable for our 3D canvas coordinates.
        # Let's return a value roughly between -2 and 2.
        
        z = (current_size - STD_SIZE) / 50.0  # value varies roughly -2 to +2
        return z


    def is_drawing_gesture(self, lm_list, threshold_cm=3.0, debug=False):
        """
        Check if index finger and thumb are close (pinched).
        threshold_cm: Distance in cm to consider as a pinch.
        """
        if len(lm_list) < 21:
            return False, (0,0,0)

        # Thumb tip (4) and Index tip (8)
        dist_cm = self.get_3d_distance_cm(lm_list, 4, 8)
        
        # Midpoint for drawing
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Z approximation
        z1, z2 = lm_list[4][3], lm_list[8][3]
        cz = (z1 + z2) / 2
        
        if debug:
            print(f"Draw Dist: {dist_cm:.2f} cm")

        if dist_cm < threshold_cm:
            return True, (cx, cy, cz)
        else:
            return False, (cx, cy, cz)

    def is_erasing_gesture(self, lm_list, threshold_cm=3.0, debug=False):
        """
        Check if pinky finger and thumb are close (pinched).
        threshold_cm: Distance in cm to consider as an erase pinch.
        """
        if len(lm_list) < 21:
            return False, (0,0,0)

        # Thumb tip (4) and Pinky tip (20)
        dist_cm = self.get_3d_distance_cm(lm_list, 4, 20)
        
        # Midpoint
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[20][1], lm_list[20][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Z approximation
        z1, z2 = lm_list[4][3], lm_list[20][3]
        cz = (z1 + z2) / 2
        
        if debug:
            print(f"Erase Dist: {dist_cm:.2f} cm")

        if dist_cm < threshold_cm:
            return True, (cx, cy, cz)
        else:
            return False, (cx, cy, cz)

    def is_hands_spread_gesture(self, img):
        """
        Check if two hands are visible and all fingers are spread open.
        Returns:
            is_spread (bool): True if both hands are spread and NOT near border
        """
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) == 2:
                # Check BOTH hands
                hand1 = self.results.multi_hand_landmarks[0]
                hand2 = self.results.multi_hand_landmarks[1]

                # Check if near border
                if self._is_near_border(hand1) or self._is_near_border(hand2):
                    return False

                hand1_open = self._is_hand_open(hand1)
                hand2_open = self._is_hand_open(hand2)
                
                if hand1_open and hand2_open:
                    return True
        return False

    def _is_near_border(self, hand_landmarks, margin=0.05):
        """
        Check if any landmark of the hand is within the margin of the image border.
        margin: percentage of width/height (0.05 = 5%)
        """
        for lm in hand_landmarks.landmark:
            if lm.x < margin or lm.x > (1 - margin) or lm.y < margin or lm.y > (1 - margin):
                return True
        return False

    def _is_hand_open(self, hand_landmarks):
        """
        Check if all fingers in a hand are extended.
        """
        # Tip IDs: 4, 8, 12, 16, 20
        # PIP IDs (Knuckle-mid): 2, 6, 10, 14, 18 or MCP: 1, 5, 9, 13, 17?
        # A simple check: Tip is further from wrist (0) than PIP (Phalange)
        
        # Fingers: Index(8), Middle(12), Ring(16), Pinky(20)
        # Compare Tip distance to Wrist vs PIP distance to Wrist
        # PIP IDs: 6, 10, 14, 18
        
        # Wrist is 0
        wrist = hand_landmarks.landmark[0]
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[pip_id]
            
            # Simple distance squared check
            dist_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
            dist_pip = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
            
            if dist_tip <= dist_pip:
                return False # Finger is curled
                
        # Thumb check: Tip(4) vs IP(3) vs MCP(2). 
        # Thumb is tricky. Vector check is better usually, but let's try distance to Pinky MCP(17) per implementation plan?
        # Or simply: Distance from ThumbTip(4) to PinkyMCP(17) should be LARGE.
        # But "Spread" implies finger extension.
        # Let's try: Thumb Tip(4) further from IndexMCP(5) than ThumbIP(3) is.
        # This usually works for "open hand" where thumb is out.
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        index_mcp = hand_landmarks.landmark[5] # Hand center-ish reference
        
        dist_thumb_tip = (thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2
        dist_thumb_ip = (thumb_ip.x - index_mcp.x)**2 + (thumb_ip.y - index_mcp.y)**2
        
        if dist_thumb_tip <= dist_thumb_ip:
             return False
             
        return True

    def is_fist_gesture(self, lm_list, debug=False):
        """
        Check if the hand is in a fist (all fingers curled).
        """
        if len(lm_list) < 21:
             return False

        # Check if 4 fingers (Index, Middle, Ring, Pinky) are curled
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18] # PIP is knuckle
        
        # Wrist is lm_list[0]
        wx, wy = lm_list[0][1], lm_list[0][2]
        
        fingers_curled = 0
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tx, ty = lm_list[tip_id][1], lm_list[tip_id][2]
            px, py = lm_list[pip_id][1], lm_list[pip_id][2]
            
            dist_tip = (tx - wx)**2 + (ty - wy)**2
            dist_pip = (px - wx)**2 + (py - wy)**2
            
            # If Tip is CLOSER to Wrist than PIP is, it's curled
            if dist_tip < dist_pip:
                fingers_curled += 1
                
        # Thumb check: 
        # To avoid "Thumbs Up" triggering rotation, we require the thumb to be curled/wrapped.
        # Robust Check: Compare Thumb Tip (4) distance to Pinky MCP (17) vs Thumb IP (3) distance to Pinky MCP (17).
        # In a fist (thumb wrapped or tucked), the Tip is closer to the Pinky base than the IP joint is.
        # In an open hand or thumbs up, the Tip is further away.
        
        thumb_tip = lm_list[4]
        thumb_ip = lm_list[3]
        pinky_mcp = lm_list[17]
        
        tx, ty = thumb_tip[1], thumb_tip[2]
        ipx, ipy = thumb_ip[1], thumb_ip[2]
        px, py = pinky_mcp[1], pinky_mcp[2]
        
        dist_tip_pinky = (tx - px)**2 + (ty - py)**2
        dist_ip_pinky = (ipx - px)**2 + (ipy - py)**2
        
        thumb_curled = dist_tip_pinky < dist_ip_pinky
        
        if debug:
            print(f"Fingers Curled: {fingers_curled}, Thumb Curled: {thumb_curled}")
            
        return fingers_curled >= 4 and thumb_curled
