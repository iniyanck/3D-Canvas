import cv2
import numpy as np

class UIOverlay:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.panel_width = 220 # Increased panel width
        self.buttons = []
        
        # Tool States
        self.active_tool = "BRUSH"
        self.manipulation_mode = "ROTATE" # ROTATE or MOVE
        self.active_shape_type = "CUBE"
        self.menu_state = "MAIN" 
        
        # Submenu State
        self.active_submenu = None # None, "BRUSH_SETTINGS", "SHAPE_SETTINGS"
        
        # Properties
        self.brush_thickness = 5
        self.brush_color = (0, 255, 0) # Green default
        self.brush_opacity = 1.0 
        

        self.shape_color = (255, 255, 255) # White default
        self.shape_opacity = 1.0

        # Eraser Settings
        self.eraser_thickness = 30

        # Selection Settings
        self.select_mode = "SINGLE" # "SINGLE" or "BOX"
        
        # UI Layout Constants
        self.x = 60
        self.y_start = 40
        self.btn_h = 45
        self.gap = 10
        self.w = 180 # Increased Main Button Width to fit text
        self.arrow_w = 40 # Arrow Button Width
        
        # Setup
        self.setup_ui()

    def setup_ui(self):
        # Tools that have submenus
        self.main_tools = ["BRUSH", "ERASER", "SELECT", "SHAPES", "UNDO", "REDO", "MANIPULATION_TOGGLE"]
        self.shape_tools = ["CUBE", "SPHERE", "PYRAMID", "BACK"] # Legacy list
        
        self.update_buttons()
        
    def update_buttons(self):
        self.buttons = []
        
        current_y = self.y_start
        tools = self.main_tools
        
        for i, tool in enumerate(tools):
            label = tool
            id = tool
            has_submenu = False
            
            if tool == "MANIPULATION_TOGGLE":
                 label = f"MODE: {self.manipulation_mode}"
            elif tool == "BRUSH":
                has_submenu = True
            elif tool == "ERASER":
                has_submenu = True
            elif tool == "SELECT":
                has_submenu = True
                label = f"SELECT: {self.select_mode}"
            elif tool == "SHAPES":
                has_submenu = True
                label = f"SHAPE: {self.active_shape_type}"
            
            btn_w = self.w
            arrow_rect = None
            
            if has_submenu:
                btn_w = self.w - self.arrow_w
                arrow_rect = (self.x + btn_w, current_y, self.arrow_w, self.btn_h)
            
            main_rect = (self.x, current_y, btn_w, self.btn_h)
            
            self.buttons.append({
                "id": id,
                "label": label,
                "rect": main_rect,
                "type": "ACTION",
                "color": (200, 200, 200)
            })
            
            if has_submenu:
                sub_id = f"{tool}_SETTINGS"
                self.buttons.append({
                    "id": sub_id,
                    "label": ">",
                    "rect": arrow_rect,
                    "type": "SUBMENU_TOGGLE",
                    "color": (150, 150, 150)
                })
            
            current_y += self.btn_h + self.gap

    def draw(self, img):
        self.update_buttons() # Refresh labels/layout
        
        # 1. Draw Main Panel Background
        total_h = len(self.main_tools) * (self.btn_h + self.gap) + self.y_start + 10
        overlay = img.copy()
        cv2.rectangle(overlay, (self.x - 10, self.y_start - 10), (self.x + self.w + 10, total_h), (50, 50, 50), -1)
        
        # 2. Draw Submenu Panel (if active)
        if self.active_submenu:
            sub_x = self.x + self.w + 20
            sub_y = self.y_start
            sub_w = 250
            sub_h = 400
            cv2.rectangle(overlay, (sub_x, sub_y), (sub_x + sub_w, sub_y + sub_h), (40, 40, 40), -1)
            
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 3. Draw Buttons
        for btn in self.buttons:
            self.draw_button(img, btn)
            
        # 4. Draw Submenu Content
        if self.active_submenu:
            self.draw_submenu(img)

    def draw_button(self, img, btn):
        x, y, w, h = btn["rect"]
        color = btn["color"]
        label = btn["label"]
        id = btn["id"]
        
        # Highlight Active
        if id == self.active_tool:
            color = (0, 255, 0)
        elif self.active_submenu and id == f"{self.active_submenu}": # Highlight Arrow if open?
             if btn["type"] == "SUBMENU_TOGGLE":
                 color = (0, 200, 255)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = x + (w - text_w) // 2
        ty = y + (h + text_h) // 2
        cv2.putText(img, label, (tx, ty), font, font_scale, (0, 0, 0), thickness)

    def draw_submenu(self, img):
        sub_x = self.x + self.w + 30
        sub_y = self.y_start + 10
        
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, f"{self.active_submenu.replace('_SETTINGS', '')}", (sub_x, sub_y + 15), font, 1.2, (255, 255, 255), 1)
        
        current_y = sub_y + 40
        
        if self.active_submenu == "BRUSH_SETTINGS":
            # Thickness Slider
            self.brush_thickness = self.draw_slider(img, "Thickness", self.brush_thickness, 1, 20, sub_x, current_y)
            current_y += 60
            
            # Color Picker
            self.brush_color = self.draw_color_picker(img, "Color", self.brush_color, sub_x, current_y)
            
        elif self.active_submenu == "SHAPES_SETTINGS":
             # Shape Types
             types = ["CUBE", "PYRAMID", "SPHERE"]
             for t in types:
                 selected = (self.active_shape_type == t)
                 col = (0, 255, 0) if selected else (100, 100, 100)
                 cv2.rectangle(img, (sub_x, current_y), (sub_x + 100, current_y + 30), col, -1)
                 cv2.putText(img, t, (sub_x + 10, current_y + 20), font, 1, (255, 255, 255), 1)
                 current_y += 40
             

             
             # Color Picker
             self.shape_color = self.draw_color_picker(img, "Color", self.shape_color, sub_x, current_y)

        elif self.active_submenu == "ERASER_SETTINGS":
             self.eraser_thickness = self.draw_slider(img, "Size", self.eraser_thickness, 10, 100, sub_x, current_y)

        elif self.active_submenu == "SELECT_SETTINGS":
             modes = ["SINGLE", "BOX"]
             for m in modes:
                 selected = (self.select_mode == m)
                 col = (0, 255, 0) if selected else (100, 100, 100)
                 cv2.rectangle(img, (sub_x, current_y), (sub_x + 100, current_y + 30), col, -1)
                 cv2.putText(img, m, (sub_x + 10, current_y + 20), font, 1, (255, 255, 255), 1)
                 current_y += 40

    def draw_slider(self, img, label, value, min_val, max_val, x, y):
        w = 200
        h = 20
        # Label
        cv2.putText(img, f"{label}: {int(value)}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
        # Bar
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), -1)
        # Handle
        norm = (value - min_val) / (max_val - min_val)
        hx = int(x + norm * w)
        cv2.circle(img, (hx, y + h//2), 10, (255, 255, 255), -1)
        return value

    def draw_color_picker(self, img, label, current_color, x, y):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (0, 255, 255), (255, 255, 0), (255, 0, 255),
            (255, 255, 255), (128, 128, 128)
        ]
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
        
        size = 30
        cols = 4
        for i, col in enumerate(colors):
            r = i // cols
            c = i % cols
            bx = x + c * (size + 5)
            by = y + r * (size + 5)
            
            # Draw in BGR
            draw_col = (col[2], col[1], col[0])
            cv2.rectangle(img, (bx, by), (bx+size, by+size), draw_col, -1)
            
            if col == current_color:
                cv2.rectangle(img, (bx, by), (bx+size, by+size), (255, 255, 255), 2)
                
        # Preview Box BGR
        if len(current_color) >= 3:
             preview = (current_color[2], current_color[1], current_color[0])
        else:
             preview = current_color
             
        cv2.rectangle(img, (x + 160, y), (x + 200, y + 40), preview, -1)
        cv2.rectangle(img, (x + 160, y), (x + 200, y + 40), (255, 255, 255), 1)
        return current_color

    def check_click(self, x, y):
        # 1. Check Submenu
        if self.active_submenu:
            sub_x = self.x + self.w + 30
            sub_y = self.y_start + 10
            
            if x > sub_x and x < sub_x + 250: 
                 current_y = sub_y + 40
                 
                 if self.active_submenu == "BRUSH_SETTINGS":
                     # Thickness
                     if current_y <= y <= current_y + 20: 
                         norm = (x - sub_x) / 200.0
                         self.brush_thickness = 1 + max(0, min(1, norm)) * 19
                         return "UPDATE_SETTINGS"
                     current_y += 60
                     # Color
                     size = 30
                     cols = 4
                     colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(255,255,255),(128,128,128)]
                     for i, col in enumerate(colors):
                        r = i//cols; c = i%cols
                        bx = sub_x + c*(size+5); by = current_y + r*(size+5)
                        if bx<=x<=bx+size and by<=y<=by+size:
                            self.brush_color = col; return "UPDATE_SETTINGS"
                            
                 elif self.active_submenu == "SHAPES_SETTINGS":
                      types = ["CUBE", "PYRAMID", "SPHERE"]
                      for t in types:
                          if sub_x <= x <= sub_x + 100 and current_y <= y <= current_y + 30:
                              self.active_shape_type = t
                              self.active_tool = "SHAPES" 
                              # Note: We DON'T close submenu here, user might want to set color too
                              return "SHAPE_TYPE_SELECTED"
                          current_y += 40

                      # Color
                      size = 30
                      cols = 4
                      colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(255,255,255),(128,128,128)]
                      for i, col in enumerate(colors):
                        r = i//cols; c = i%cols
                        bx = sub_x + c*(size+5); by = current_y + r*(size+5)
                        if bx<=x<=bx+size and by<=y<=by+size:
                            self.shape_color = col; return "UPDATE_SETTINGS"

                 elif self.active_submenu == "ERASER_SETTINGS":
                       if current_y <= y <= current_y + 20:
                            norm = (x - sub_x) / 200.0
                            self.eraser_thickness = 10 + max(0, min(1, norm)) * 90
                            return "UPDATE_SETTINGS"

                 elif self.active_submenu == "SELECT_SETTINGS":
                      modes = ["SINGLE", "BOX"]
                      for m in modes:
                          if sub_x <= x <= sub_x + 100 and current_y <= y <= current_y + 30:
                              self.select_mode = m
                              self.active_tool = "SELECT"
                              return "SELECT_MODE_SELECTED"
                          current_y += 40

        # 2. Check Main Buttons
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                id = btn["id"]
                
                if btn["type"] == "SUBMENU_TOGGLE":
                    target_sub = id
                    if self.active_submenu == target_sub:
                        self.active_submenu = None # Toggle Off
                    else:
                        self.active_submenu = target_sub # Switch
                    return "SUBMENU_TOGGLED"
                
                elif btn["type"] == "ACTION":
                    if id == "MANIPULATION_TOGGLE":
                        self.toggle_manipulation_mode()
                        return "MANIPULATION_TOGGLE"
                    elif id in ["UNDO", "REDO"]:
                        # Global actions don't change the active tool
                        return id
                    else:
                        self.active_tool = id
                        # Auto-Close Submenu if switching to a different tool concept
                        # If I click BRUSH (Main), and SHAPES_SETTINGS was open, CLOSE IT.
                        # If I click BRUSH (Main), and BRUSH_SETTINGS was open, keep it? Or toggle?
                        # User said: "make the submenus go away when selecting another thing from the main menu"
                        
                        # So if I click ANYTHING in main menu, close submenu UNLESS it's the toggle interaction.
                        # We are in ACTION handler here.
                        
                        # If I click "BRUSH", I want to select brush. If "SHAPES" menu was open, close it.
                        self.active_submenu = None 
                        
                        return id
        
        return None

    def toggle_manipulation_mode(self):
        if self.manipulation_mode == "ROTATE":
            self.manipulation_mode = "MOVE"
        else:
            self.manipulation_mode = "ROTATE" 
 

