import cv2
import numpy as np

class UIOverlay:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.panel_width = 100
        self.buttons = []
        self.active_tool = "BRUSH"
        self.sub_options_visible = False
        
        # Tools
        self.setup_ui()

    def setup_ui(self):
        # Define Buttons: (x, y, w, h, ID, Label, Color)
        # Vertical Layout on the Left
        y_start = 50
        btn_h = 60
        gap = 10
        x = 10
        w = 80
        
        tools = ["BRUSH", "ERASER", "SELECT", "SHAPES", "UNDO", "REDO"]
        
        for i, tool in enumerate(tools):
            self.buttons.append({
                "rect": (x, y_start + i*(btn_h+gap), w, btn_h),
                "id": tool,
                "label": tool,
                "color": (200, 200, 200)
            })

    def draw(self, img):
        # Draw Panel Background
        # Just a visual separation is enough or a semi-transparent rect
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.panel_width, self.height), (50, 50, 50), -1)
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw Buttons
        for btn in self.buttons:
            x, y, w, h = btn["rect"]
            color = btn["color"]
            
            # Highlight Active Tool
            if btn["id"] == self.active_tool:
                color = (0, 255, 0) # Green for active
            elif btn["id"] in ["UNDO", "REDO"]:
                color = (100, 100, 255) # Reddish for actions

            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            # Text
            # Calculate center roughly
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(btn["label"], font, font_scale, thickness)
            
            tx = x + (w - text_w) // 2
            ty = y + (h + text_h) // 2
            
            cv2.putText(img, btn["label"], (tx, ty), font, font_scale, (0, 0, 0), thickness)

    def check_click(self, x, y):
        """
        Returns the ID of the button clicked, or None.
        Updates active_tool if it's a tool button.
        """
        # Check if click is inside panel area
        if x > self.panel_width:
            return None
            
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                # Handle Tool Switching
                if btn["id"] in ["BRUSH", "ERASER", "SELECT", "SHAPES"]:
                    self.active_tool = btn["id"]
                return btn["id"]
        
        return None
