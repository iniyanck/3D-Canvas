import cv2
import numpy as np

class UIOverlay:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.panel_width = 210 # Increased to cover buttons at x=100 + w=80 + padding
        self.buttons = []
        self.active_tool = "BRUSH"
        self.manipulation_mode = "ROTATE" # ROTATE or MOVE
        self.active_shape_type = "CUBE"
        self.menu_state = "MAIN" 
        self.sub_options_visible = False # Deprecated but keeping for safety
        self.sub_buttons = []
        
        # Tools
        self.setup_ui()

    def setup_ui(self):
        # We will regenerate buttons on demand or store them in separate lists
        # Let's use separate lists and switch self.buttons pointer or logic
        
        # Common layout
        self.y_start = 40
        self.btn_h = 45
        self.gap = 5
        self.x = 60
        self.w = 150
        
        self.main_tools = ["BRUSH", "ERASER", "SELECT", "SHAPES", "UNDO", "REDO", "MANIPULATION_TOGGLE"]
        self.shape_tools = ["CUBE", "SPHERE", "PYRAMID", "BACK"]
        
        self.update_buttons()
        
    def update_buttons(self):
        self.buttons = []
        tools = self.main_tools if self.menu_state == "MAIN" else self.shape_tools
        
        for i, tool in enumerate(tools):
            label = tool
            id = tool
            
            if tool == "MANIPULATION_TOGGLE":
                 label = f"MODE: {self.manipulation_mode}"
            elif tool in ["CUBE", "SPHERE", "PYRAMID"]:
                 id = f"SHAPE_{tool}"
                 if id.replace("SHAPE_", "") == self.active_shape_type and self.menu_state == "SHAPES":
                     label = f"> {tool} <" # Highlight
            
            self.buttons.append({
                "rect": (self.x, self.y_start + i*(self.btn_h+self.gap), self.w, self.btn_h),
                "id": id,
                "label": label,
                "color": (200, 200, 200)
            })
            
    def draw(self, img):
        # Refresh buttons just in case state changed (simplistic)
        self.update_buttons() # Or only update on state change
        
        # Draw Panel Background (dynamic height based on buttons)
        h = len(self.buttons) * (self.btn_h + self.gap) + self.y_start + 20
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.x + self.w + 20, h), (50, 50, 50), -1)
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw Buttons
        for btn in self.buttons:
            x, y, w, h = btn["rect"]
            color = btn["color"]
            
            # Highlight Active Tool
            if self.menu_state == "MAIN":
                if btn["id"] == self.active_tool:
                    color = (0, 255, 0)
            elif self.menu_state == "SHAPES":
                if btn["id"].replace("SHAPE_", "") == self.active_shape_type and btn["id"] != "BACK":
                     color = (0, 255, 0)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
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
        # Check Panel Buttons
        clicked_id = None
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                clicked_id = btn["id"]
                break
        
        if not clicked_id:
            return None
            
        if clicked_id == "SHAPES":
            self.menu_state = "SHAPES"
            self.active_tool = "SHAPES"
            self.update_buttons() # Immediate update
            return "MENU_CHANGED"
            
        elif clicked_id == "BACK":
            self.menu_state = "MAIN"
            self.active_tool = "BRUSH" # Default back to brush?
            self.update_buttons()
            return "MENU_CHANGED"
            
        elif clicked_id == "MANIPULATION_TOGGLE":
            self.toggle_manipulation_mode()
            return "MANIPULATION_TOGGLE"
            
        elif clicked_id.startswith("SHAPE_"):
            self.active_shape_type = clicked_id.replace("SHAPE_", "")
            self.update_buttons()
            return "SHAPE_TYPE_SELECTED"
            
        else:
            self.active_tool = clicked_id
            return clicked_id

    def toggle_manipulation_mode(self):
        if self.manipulation_mode == "ROTATE":
            self.manipulation_mode = "MOVE"
        else:
            self.manipulation_mode = "ROTATE" 
        self.update_buttons()
