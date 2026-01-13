import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

class Canvas3D:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        # self.lines will now store dicts: {"points": [(x,y,z),...], "color": (r,g,b), "thickness": t}
        self.lines = [] 
        self.shapes = [] # List of shape dicts: {type, bounds: (p1, p2), color}
        self.current_stroke = []
        self.cursor_pos = None # (x, y, z) tuple for 3D cursor
        
        # Current Settings
        self.current_color = (1.0, 1.0, 1.0) # Default White (R,G,B 0-1)
        self.current_thickness = 1
        
        # Tool State
        self.selected_indices = [] # Indices of selected strokes in self.lines
        self.selected_shape_indices = [] # Indices of selected shapes in self.shapes
        self.undo_stack = []
        self.redo_stack = []
        
        # Rotation State
        self.rot_x = 0.0 # Pitch
        self.rot_y = 0.0 # Yaw
        
        # Camera State
        self.camera_z = -3.0 # Distance from pivot
        self.target_x = 0.0 # Pivot Point X
        self.target_y = 0.0 # Pivot Point Y
        self.target_z = 0.0 # Pivot Point Z

        # Quadric
        self.quadric = gluNewQuadric()

        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Gesture Canvas")

        # Basic OpenGL Setup
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, self.camera_z) 
        
        # Lighting Setup
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_LIGHTING)   
        glEnable(GL_COLOR_MATERIAL) 
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light 0: Static
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 10, 10, 0]) 
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Light 1: Dynamic Cursor
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 1.0, 0.8, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [1.0, 1.0, 1.0, 1])
        glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.2)
        glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.02)

    def set_color(self, rgb_255):
        """Set current RGB color (input tuple 0-255)"""
        # Ensure we have a tuple of 3
        if len(rgb_255) >= 3:
            self.current_color = (rgb_255[0]/255.0, rgb_255[1]/255.0, rgb_255[2]/255.0)

    def set_thickness(self, val):
        self.current_thickness = val

    def save_state(self):
        import copy
        self.undo_stack.append({
            "lines": copy.deepcopy(self.lines),
            "shapes": copy.deepcopy(self.shapes)
        })
        if len(self.undo_stack) > 20: self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            state = self.undo_stack.pop()
            # Push current to Redo
            self.redo_stack.append({
                "lines": self.lines,
                "shapes": self.shapes
            })
            self.lines = state["lines"]
            self.shapes = state["shapes"]
            self.selected_indices = [] 

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append({
                "lines": self.lines,
                "shapes": self.shapes
            })
            self.lines = state["lines"]
            self.shapes = state["shapes"]
            self.selected_indices = []

    def get_world_point_from_view(self, x, y, z_relative):
        # 1. NDC
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1)
        
        # 2. View Space
        fov = 45
        aspect = self.width / self.height
        tan_half_fov = math.tan(math.radians(fov / 2))
        
        DRAWING_DIST = -self.camera_z 
        depth = DRAWING_DIST - z_relative * 1.0 
        depth = max(0.5, depth)
        
        x_view = norm_x * depth * aspect * tan_half_fov
        y_view = norm_y * depth * tan_half_fov
        z_view = -depth
        
        p_view = np.array([x_view, y_view, z_view])
        
        # 3. View -> World
        p_temp = p_view - np.array([0, 0, self.camera_z])
        
        rad_x = np.radians(self.rot_x)
        rad_y = np.radians(self.rot_y)
        
        rx_mat = np.array([
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x), np.cos(rad_x)]
        ])
        
        ry_mat = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])
        
        inv_rx = rx_mat.T
        inv_ry = ry_mat.T
        
        p_temp = np.dot(inv_rx, p_temp)
        p_temp = np.dot(inv_ry, p_temp)
        
        p_world = p_temp + np.array([self.target_x, self.target_y, self.target_z])
        
        return tuple(p_world)

    def add_point(self, x, y, z, start_new_stroke=False):
        world_point = self.get_world_point_from_view(x, y, z)

        if start_new_stroke:
            self.end_stroke() # Commit previous if exists
                
        MIN_DIST = 0.02
        if self.current_stroke:
            last_p = np.array(self.current_stroke[-1])
            new_p = np.array(world_point)
            if np.linalg.norm(new_p - last_p) < MIN_DIST:
                return 
        
        self.current_stroke.append(world_point)

    def end_stroke(self):
        if self.current_stroke:
            self.save_state()
            # Store as dict
            self.lines.append({
                "points": self.current_stroke,
                "color": self.current_color,
                "thickness": self.current_thickness
            })
            self.current_stroke = []

    def update_cursor(self, x, y, z):
        self.cursor_pos = self.get_world_point_from_view(x, y, z)

    def clear(self):
        self.save_state()
        self.lines = []
        self.shapes = []
        self.current_stroke = []
        self.selected_indices = []
        self.selected_shape_indices = []

    def get_ray_from_screen(self, screen_x, screen_y):
        # 1. NDC
        norm_x = (screen_x / self.width) * 2 - 1
        norm_y = -((screen_y / self.height) * 2 - 1)
        
        # 2. View Space
        fov = 45
        aspect = self.width / self.height
        tan_half_fov = math.tan(math.radians(fov / 2))
        
        vx = norm_x * aspect * tan_half_fov
        vy = norm_y * tan_half_fov
        vz = -1.0
        
        dir_view = np.array([vx, vy, vz])
        dir_view = dir_view / np.linalg.norm(dir_view)
        
        # 3. View -> World
        # Inverse Rotation
        rad_x = np.radians(self.rot_x)
        rad_y = np.radians(self.rot_y)
        
        rx_mat = np.array([
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x), np.cos(rad_x)]
        ])
        ry_mat = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])
        
        inv_rx = rx_mat.T
        inv_ry = ry_mat.T
        
        # Ray Origin View (0,0,-cam_z) relative to pivot ??
        # The logic in get_world_point is:
        # p_temp = p_view - [0, 0, cam_z]
        # p_world = InvRot(p_temp) + target
        
        p_origin_view = np.array([0, 0, 0])
        p_temp = p_origin_view - np.array([0, 0, self.camera_z]) # (0,0, -cam_z)
        
        d_temp = dir_view
        
        p_temp = np.dot(inv_rx, p_temp)
        p_temp = np.dot(inv_ry, p_temp)
        
        d_temp = np.dot(inv_rx, d_temp)
        d_temp = np.dot(inv_ry, d_temp)
        
        ray_origin = p_temp + np.array([self.target_x, self.target_y, self.target_z])
        ray_dir = d_temp
        
        return ray_origin, ray_dir

    def ray_intersects_aabb(self, ray_o, ray_d, box_min, box_max):
        # Slab method
        t_min = -1e9
        t_max = 1e9
        
        for i in range(3):
            if abs(ray_d[i]) < 1e-6:
                if ray_o[i] < box_min[i] or ray_o[i] > box_max[i]:
                    return False
            else:
                t1 = (box_min[i] - ray_o[i]) / ray_d[i]
                t2 = (box_max[i] - ray_o[i]) / ray_d[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
        
        # Check within ray limits (t > 0)
        return t_max >= t_min and t_max > 0

    def clear_selection(self):
        self.selected_indices = []
        self.selected_shape_indices = []

    def select_at(self, screen_x, screen_y, radius=30):
        # Robust Selection: Uses Screen-Space Projection + Depth Sorting
        ray_o, ray_d = self.get_ray_from_screen(screen_x, screen_y)
        
        candidates = [] # List of (depth, type, index)
        
        # 1. Check Lines (Distance < Radius)
        cursor_pos = np.array([screen_x, screen_y])
        for i, stroke_data in enumerate(self.lines):
            stroke = stroke_data["points"]
            # Check every 5th point for speed
            for p3d in stroke[::5]: 
                p2d = self.project_point(p3d)
                if p2d:
                    dist = np.linalg.norm(np.array(p2d) - cursor_pos)
                    if dist < radius:
                        # Hit! Record depth relative to camera
                        depth = np.linalg.norm(np.array(p3d) - ray_o)
                        candidates.append((depth, "LINE", i))
                        break # Found a hit for this stroke
        
        # 2. Check Shapes (Screen Space AABB)
        for i, shape in enumerate(self.shapes):
            center = shape.get("center")
            size = shape.get("size")
            # Get rotation matrix, default to identity
            rot_mat = shape.get("rotation_matrix", np.identity(3))
            
            if center is not None and size is not None:
                # Generate 8 corners in Local Space
                dx, dy, dz = size
                corners_local = []
                for sx in [-0.5, 0.5]:
                     for sy in [-0.5, 0.5]:
                          for sz in [-0.5, 0.5]:
                               corners_local.append(np.array([sx*dx, sy*dy, sz*dz]))
                
                valid_2d = []
                for p in corners_local:
                    # Apply Rotation Matrix
                    # p is (3,)
                    p_rot = np.dot(rot_mat, p)
                    
                    # Translate
                    pt_world = p_rot + center
                    
                    p2d = self.project_point(pt_world)
                    if p2d:
                        valid_2d.append(p2d)
                
                if valid_2d:
                    pts = np.array(valid_2d)
                    min_x, min_y = np.min(pts, axis=0)
                    max_x, max_y = np.max(pts, axis=0)
                    
                    pad = 10
                    # Check if cursor is inside 2D projected AABB
                    if (min_x - pad <= screen_x <= max_x + pad) and (min_y - pad <= screen_y <= max_y + pad):
                        dist_to_cam = np.linalg.norm(center - ray_o)
                        candidates.append((dist_to_cam, "SHAPE", i))

        # Decision: Pick closest
        if candidates:
            candidates.sort(key=lambda x: x[0]) # Sort by depth
            _, type, idx = candidates[0]
            
            if type == "LINE":
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices = [idx]
                self.selected_shape_indices = [] # Exclusive
            else:
                if idx in self.selected_shape_indices:
                    self.selected_shape_indices.remove(idx)
                else:
                    self.selected_shape_indices = [idx]
                self.selected_indices = [] # Exclusive
        else:
            self.selected_indices = []
            self.selected_shape_indices = []

    def select_in_volume(self, p1, p2, add_to_selection=False):
        # Create AABB from p1, p2
        center = (np.array(p1) + np.array(p2)) / 2.0
        world_diag = np.array(p2) - np.array(p1)
        size = np.abs(world_diag)
        
        # Simple AABB logic for now (Axial Only):
        min_pt = center - size / 2.0
        max_pt = center + size / 2.0
        
        if not add_to_selection:
            self.selected_indices = []
            self.selected_shape_indices = []
        
        # 1. Select Strokes
        for i, stroke_data in enumerate(self.lines):
            stroke = stroke_data["points"]
            # Check if any point is in AABB
            found = False
            for p in stroke[::2]:
                # Check point in AABB
                if np.all(p >= min_pt) and np.all(p <= max_pt):
                    if i not in self.selected_indices:
                        self.selected_indices.append(i)
                    found = True
                    break 
            
        # 2. Select Shapes
        for i, shape in enumerate(self.shapes):
            shape_center = shape["center"]
            if np.all(shape_center >= min_pt) and np.all(shape_center <= max_pt):
                if i not in self.selected_shape_indices:
                    self.selected_shape_indices.append(i)

    def select_in_rect(self, start_x, start_y, end_x, end_y):
        x_min = min(start_x, end_x)
        x_max = max(start_x, end_x)
        y_min = min(start_y, end_y)
        y_max = max(start_y, end_y)
        
        # 1. Select Strokes
        for i, stroke_data in enumerate(self.lines):
            stroke = stroke_data["points"]
            # Check if any point is in the box
            # To speed up, check every 5th point
            found = False
            for p in stroke[::2]:
                px, py = self.project_point(p)
                if px is not None:
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        self.selected_indices.append(i)
                        found = True
                        break 
            
        # 2. Select Shapes
        for i, shape in enumerate(self.shapes):
            center = shape["center"]
            cx, cy = self.project_point(center)
            if cx is not None:
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    self.selected_shape_indices.append(i)

    def erase_at(self, screen_x, screen_y, radius=30):
        self.save_state()
        eraser_pos = np.array([screen_x, screen_y])
        new_lines = []
        
        if self.current_stroke:
            self.lines.append({
                "points": self.current_stroke,
                "color": self.current_color,
                "thickness": self.current_thickness
            })
            self.current_stroke = []

        for i, stroke_data in enumerate(self.lines):
            stroke = stroke_data["points"]
            base_color = stroke_data["color"]
            thickness = stroke_data.get("thickness", 1)
            
            current_segment = []
            
            for p3d in stroke:
                p2d = self.project_point(p3d)
                keep = True
                if p2d:
                    if np.linalg.norm(np.array(p2d) - eraser_pos) <= radius:
                        keep = False
                
                if keep:
                    current_segment.append(p3d)
                else:
                    if len(current_segment) > 1:
                        new_lines.append({
                            "points": current_segment, 
                            "color": base_color,
                            "thickness": thickness
                        })
                    current_segment = []
            
            if len(current_segment) > 1:
                new_lines.append({
                    "points": current_segment, 
                    "color": base_color,
                    "thickness": thickness
                })
                
        self.lines = new_lines
        
        # Erase Shapes
        new_shapes = []
        for shape in self.shapes:
             keep = True
             if "center" in shape:
                 center = shape["center"]
                 p2d = self.project_point(center)
                 if p2d and np.linalg.norm(np.array(p2d) - eraser_pos) < radius + 10:
                     keep = False
             if keep: new_shapes.append(shape)
        self.shapes = new_shapes
        self.selected_indices = []
        self.selected_shape_indices = []

    def project_point(self, point_world):
        x, y, z = point_world
        tx, ty, tz = x - self.target_x, y - self.target_y, z - self.target_z
        
        rad_x = np.radians(self.rot_x)
        rad_y = np.radians(self.rot_y)
        
        # Y Rot
        rx, rz = tx * np.cos(rad_y) + tz * np.sin(rad_y), -tx * np.sin(rad_y) + tz * np.cos(rad_y)
        # X Rot
        ry, rz = ty * np.cos(rad_x) - rz * np.sin(rad_x), ty * np.sin(rad_x) + rz * np.cos(rad_x)
        
        rz += self.camera_z
        
        if rz >= -0.1: return None # Behind camera
            
        fov = 45
        aspect = self.width / self.height
        f = 1.0 / math.tan(math.radians(fov) / 2)
        
        x_ndc = (f / aspect) * rx / -rz
        y_ndc = f * ry / -rz
        
        return (x_ndc + 1) * self.width / 2, (1 - y_ndc) * self.height / 2
        
    def rotate_selection(self, d_yaw, d_pitch):
        if not self.selected_indices and not self.selected_shape_indices: return
        
        # 1. Construct Screen-Space Rotation Matrix (R_inc_screen)
        # Yaw (Screen X) -> Rotate around Y
        # Pitch (Screen Y) -> Rotate around X
        rad_pitch, rad_yaw = np.radians(d_pitch), np.radians(d_yaw)
        
        # Small angle approximation or separate matrices
        mat_pitch = np.array([[1,0,0],[0,np.cos(rad_pitch),-np.sin(rad_pitch)],[0,np.sin(rad_pitch),np.cos(rad_pitch)]])
        mat_yaw = np.array([[np.cos(rad_yaw),0,np.sin(rad_yaw)],[0,1,0],[-np.sin(rad_yaw),0,np.cos(rad_yaw)]])
        
        r_inc_screen = np.dot(mat_pitch, mat_yaw)
        
        # 2. Construct View Rotation Matrix (World -> View)
        # Matches render(): glRotate(rot_x), glRotate(rot_y) -> R_view = Rx * Ry
        cam_rx = np.radians(self.rot_x)
        cam_ry = np.radians(self.rot_y)
        
        v_rx = np.array([[1,0,0],[0,np.cos(cam_rx),-np.sin(cam_rx)],[0,np.sin(cam_rx),np.cos(cam_rx)]])
        v_ry = np.array([[np.cos(cam_ry),0,np.sin(cam_ry)],[0,1,0],[-np.sin(cam_ry),0,np.cos(cam_ry)]])
        
        r_view = np.dot(v_rx, v_ry)
        
        # 3. Transform Increment to World Space
        # R_global = Inv(R_view) * R_screen * R_view
        # Since these are rotation matrices, Inv(R) = Transpose(R)
        r_view_inv = r_view.T
        
        r_inc_global = np.dot(r_view_inv, np.dot(r_inc_screen, r_view))

        # 1. Lines
        if self.selected_indices:
            # Centroid
            points = []
            for idx in self.selected_indices:
                points.extend(self.lines[idx]["points"])
            if points:
                centroid = np.mean(points, axis=0)
                
                for idx in self.selected_indices:
                    stroke = self.lines[idx]["points"]
                    new_stroke = []
                    for p in stroke:
                        v = np.array(p) - centroid
                        v = np.dot(r_inc_global, v) 
                        new_stroke.append(tuple(v + centroid))
                    self.lines[idx]["points"] = new_stroke

        # 2. Shapes (Matrix Multiplication)
        for idx in self.selected_shape_indices:
            shape = self.shapes[idx]
            current_mat = shape.get("rotation_matrix", np.identity(3))
            
            # Apply global rotation
            new_mat = np.dot(r_inc_global, current_mat)
            
            shape["rotation_matrix"] = new_mat

    def move_selection(self, dx, dy, dz):
        if not self.selected_indices and not self.selected_shape_indices: return

        # 1. Calculate Centroid Depth for Scaling
        # We want the object to "stick" to the pointer.
        # Screen Delta -> World Delta depends on Z distance.
        # Approx: WorldMove = ScreenMove * (Z / Focal)
        
        centroid = np.array([0.0, 0.0, 0.0])
        count = 0
        
        if self.selected_indices:
             for idx in self.selected_indices:
                 pts = self.lines[idx]["points"]
                 centroid += np.mean(pts, axis=0)
                 count += 1
        
        if self.selected_shape_indices:
             for idx in self.selected_shape_indices:
                 centroid += self.shapes[idx]["center"]
                 count += 1
        
        if count > 0:
            centroid /= count
        
        # Calculate Depth (Distance from Camera Plane)
        # Camera is at (target_x, target_y, camera_z) roughly, 
        # but actually we move the world by (-tx, -ty, -cz).
        # In Camera Space (View Space), the camera is at origin.
        # We need to project centroid to View Space to get Z.
        
        # Construct View Matrix
        cam_rx, cam_ry = np.radians(self.rot_x), np.radians(self.rot_y)
        v_rx = np.array([[1,0,0],[0,np.cos(cam_rx),-np.sin(cam_rx)],[0,np.sin(cam_rx),np.cos(cam_rx)]])
        v_ry = np.array([[np.cos(cam_ry),0,np.sin(cam_ry)],[0,1,0],[-np.sin(cam_ry),0,np.cos(cam_ry)]])
        r_view = np.dot(v_rx, v_ry)
        
        # Transform Centroid to View Space relative to "Camera Center"
        # World -> View is: R * (P - CameraPos)
        # But our simple camera model calculates ViewPos as:
        # R * (P - Target) + [0, 0, CamZ]
        
        target = np.array([self.target_x, self.target_y, self.target_z])
        p_rel = centroid - target
        p_view = np.dot(r_view, p_rel)
        p_view[2] += self.camera_z
        
        depth = abs(p_view[2])
        if depth < 0.1: depth = 0.1
        
        # Empirically, scale roughly by depth * constant
        # Normal depth ~ 5.0 -> scale ~ 0.01 per pixel
        scale_factor = depth * 0.002 # Tune this
        
        # 2. Transform Screen Movement to World Movement
        # Screen Move (dx, dy) is in Camera X/Y axes.
        # World Move = Inv(R_view) * [dx, dy, 0]
        
        # We use -dy because Screen Y is Down, World Y is Up
        move_view = np.array([dx, -dy, dz * -150.0]) * scale_factor
        
        r_inv = r_view.T
        move_world = np.dot(r_inv, move_view)
        
        # Lines
        for idx in self.selected_indices:
            stroke = self.lines[idx]["points"]
            new_stroke = [tuple(np.array(p) + move_world) for p in stroke]
            self.lines[idx]["points"] = new_stroke
            
        # Shapes
        for idx in self.selected_shape_indices:
            self.shapes[idx]["center"] = self.shapes[idx]["center"] + move_world

    def pan_camera(self, dx, dy, dz):
        scale_xy, scale_z = 0.02, 1.0
        pan_vec = np.array([-dx * scale_xy, dy * scale_xy, 0])
        
        rad_x, rad_y = np.radians(-self.rot_x), np.radians(-self.rot_y)
        rx_mat = np.array([[1,0,0],[0,np.cos(rad_x),-np.sin(rad_x)],[0,np.sin(rad_x),np.cos(rad_x)]])
        ry_mat = np.array([[np.cos(rad_y),0,np.sin(rad_y)],[0,1,0],[-np.sin(rad_y),0,np.cos(rad_y)]])
        
        pan_vec = np.dot(rx_mat, pan_vec)
        pan_vec = np.dot(ry_mat, pan_vec)
        
        self.target_x += pan_vec[0]
        self.target_y += pan_vec[1]
        self.target_z += pan_vec[2]
        self.camera_z = max(-20.0, min(-0.5, self.camera_z - dz * scale_z))

    def get_camera_position_world(self):
        # Cam at (0,0,0) in View Space
        # v_world = T(target) * Inv(Ry) * Inv(Rx) * Inv(T(z)) * v_view
        
        # 1. Inverse Camera Z Translation
        p = np.array([0.0, 0.0, -self.camera_z])
        
        # 2. Inverse Rx
        rad_x = np.radians(-self.rot_x)
        rx_mat = np.array([
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x), np.cos(rad_x)]
        ])
        p = np.dot(rx_mat, p)
        
        # 3. Inverse Ry
        rad_y = np.radians(-self.rot_y)
        ry_mat = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])
        p = np.dot(ry_mat, p)
        
        # 4. Inverse Target Translation
        p += np.array([self.target_x, self.target_y, self.target_z])
        
        return p

    def rotate(self, d_yaw, d_pitch):
        self.rot_y += d_yaw
        self.rot_x += d_pitch

    # --- SHAPES ---
    
    # --- SHAPES ---
    
    def _create_shape_dict(self, p1, p2, shape_type, rotation=0.0):
        # Convert Bounds -> Center + Size + Rotation
        # p1, p2 are world points of the drag diagonal
        # We assume the drag defines the size in the VIEW, but we map it to WORLD size
        # Actually simplest is: Center = midpoint, Size = abs(diff)
        
        center = (np.array(p1) + np.array(p2)) / 2.0
        
        # Initial Rotation Matrix just for the Drag Rotation (Y axis)
        # We can construct it.
        rad = np.radians(rotation)
        c, s = np.cos(rad), np.sin(rad)
        # Ry
        rot_mat = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        
        # Calculate size in local coordinates (unrotated)
        # We need to un-rotate the world-diag to get local size
        world_diag = np.array(p2) - np.array(p1)
        
        # Inverse rotate
        inv_rot = rot_mat.T 
        local_diag = np.dot(inv_rot, world_diag)
        size = np.abs(local_diag)
        
        return {
            "type": shape_type,
            "center": center,
            "size": size,
            "color": self.current_color,
            "thickness": self.current_thickness,
            "rotation_matrix": rot_mat
        }

    def add_shape_bounds(self, p1, p2, shape_type, rotation=0.0):
        self.save_state()
        shape = self._create_shape_dict(p1, p2, shape_type, rotation)
        self.shapes.append(shape)

    def preview_shape_bounds(self, p1, p2, shape_type, rotation=0.0):
        self.current_preview_shape = self._create_shape_dict(p1, p2, shape_type, rotation)

    def draw_wireframe_cube_bounds(self, p1, p2, color):
        pass # Not used

    def draw_wireframe_cube_local(self, color):
        glPushMatrix()
        glColor3f(*color[:3])
        glLineWidth(2.0)
        glBegin(GL_LINES)
        pts = [
            (-0.5,-0.5,-0.5), (0.5,-0.5,-0.5), (0.5,0.5,-0.5), (-0.5,0.5,-0.5),
            (-0.5,-0.5,0.5), (0.5,-0.5,0.5), (0.5,0.5,0.5), (-0.5,0.5,0.5)
        ]
        # Bottom Loop
        glVertex3f(*pts[0]); glVertex3f(*pts[1])
        glVertex3f(*pts[1]); glVertex3f(*pts[2])
        glVertex3f(*pts[2]); glVertex3f(*pts[3])
        glVertex3f(*pts[3]); glVertex3f(*pts[0])
        # Top Loop
        glVertex3f(*pts[4]); glVertex3f(*pts[5])
        glVertex3f(*pts[5]); glVertex3f(*pts[6])
        glVertex3f(*pts[6]); glVertex3f(*pts[7])
        glVertex3f(*pts[7]); glVertex3f(*pts[4])
        # Columns
        glVertex3f(*pts[0]); glVertex3f(*pts[4])
        glVertex3f(*pts[1]); glVertex3f(*pts[5])
        glVertex3f(*pts[2]); glVertex3f(*pts[6])
        glVertex3f(*pts[3]); glVertex3f(*pts[7])
        glEnd()
        glLineWidth(1.0)
        glPopMatrix()

    def draw_wireframe_pyramid_local(self, color):
        glPushMatrix()
        glColor3f(*color[:3])
        glLineWidth(2.0)
        glBegin(GL_LINES)
        y_base = -0.5
        pts = [(-0.5,y_base,-0.5), (0.5,y_base,-0.5), (0.5,y_base,0.5), (-0.5,y_base,0.5)]
        
        glVertex3f(*pts[0]); glVertex3f(*pts[1])
        glVertex3f(*pts[1]); glVertex3f(*pts[2])
        glVertex3f(*pts[2]); glVertex3f(*pts[3])
        glVertex3f(*pts[3]); glVertex3f(*pts[0])
        
        tip = (0.0, 0.5, 0.0)
        for p in pts:
            glVertex3f(*p); glVertex3f(*tip)
            
        glEnd()
        glLineWidth(1.0)
        glPopMatrix()
        
    def draw_solid_cube_local(self, color):
        glPushMatrix()
        if len(color) == 4:
            glColor4f(*color)
        else:
            glColor3f(*color)
            
        glBegin(GL_QUADS)
        glNormal3f(0,0,1); glVertex3f(-0.5,-0.5,0.5); glVertex3f(0.5,-0.5,0.5); glVertex3f(0.5,0.5,0.5); glVertex3f(-0.5,0.5,0.5)
        glNormal3f(0,0,-1); glVertex3f(-0.5,-0.5,-0.5); glVertex3f(-0.5,0.5,-0.5); glVertex3f(0.5,0.5,-0.5); glVertex3f(0.5,-0.5,-0.5)
        glNormal3f(0,1,0); glVertex3f(-0.5,0.5,-0.5); glVertex3f(-0.5,0.5,0.5); glVertex3f(0.5,0.5,0.5); glVertex3f(0.5,0.5,-0.5)
        glNormal3f(0,-1,0); glVertex3f(-0.5,-0.5,-0.5); glVertex3f(0.5,-0.5,-0.5); glVertex3f(0.5,-0.5,0.5); glVertex3f(-0.5,-0.5,0.5)
        glNormal3f(1,0,0); glVertex3f(0.5,-0.5,-0.5); glVertex3f(0.5,0.5,-0.5); glVertex3f(0.5,0.5,0.5); glVertex3f(0.5,-0.5,0.5)
        glNormal3f(-1,0,0); glVertex3f(-0.5,-0.5,-0.5); glVertex3f(-0.5,-0.5,0.5); glVertex3f(-0.5,0.5,0.5); glVertex3f(-0.5,0.5,-0.5)
        glEnd()
        glPopMatrix()

    def draw_solid_pyramid_local(self, color):
        glPushMatrix()
        if len(color) == 4:
            glColor4f(*color)
        else:
            glColor3f(*color)
        
        tip = (0, 0.5, 0)
        y_base = -0.5
        p1=(-0.5,y_base,-0.5); p2=(0.5,y_base,-0.5); p3=(0.5,y_base,0.5); p4=(-0.5,y_base,0.5)
        
        glBegin(GL_TRIANGLES)
        glNormal3f(0, 0.5, 1); glVertex3f(*tip); glVertex3f(*p4); glVertex3f(*p3)
        glNormal3f(1, 0.5, 0); glVertex3f(*tip); glVertex3f(*p3); glVertex3f(*p2)
        glNormal3f(0, 0.5, -1); glVertex3f(*tip); glVertex3f(*p2); glVertex3f(*p1)
        glNormal3f(-1, 0.5, 0); glVertex3f(*tip); glVertex3f(*p1); glVertex3f(*p4)
        glEnd()
        
        glBegin(GL_QUADS)
        glNormal3f(0, -1, 0)
        glVertex3f(*p4); glVertex3f(*p1); glVertex3f(*p2); glVertex3f(*p3)
        glEnd()
        glPopMatrix()

    def render_shape(self, shape_dict, is_ghost=False, is_selected=False):
        center = shape_dict["center"]
        size = shape_dict["size"]
        rot_mat = shape_dict.get("rotation_matrix", np.identity(3))
        
        color = shape_dict.get("color", (1,1,1))
        type = shape_dict.get("type", "CUBE")
        # Removing smoothness param, default high segments
        segments = 32
        
        glPushMatrix()
        glTranslatef(*center)
        
        # Apply Rotation Matrix
        m_4x4 = np.identity(4)
        m_4x4[:3, :3] = rot_mat.T 
        glMultMatrixf(m_4x4.flatten())
        
        glScalef(*size)
        
        if is_ghost:
             self.draw_wireframe_cube_local((1,1,1))
             r,g,b = color if len(color)==3 else (1,1,1)
             rgba = (r, g, b, 0.4) 
             if type == "SELECTION_BOX":
                 # Green Wireframe only
                 self.draw_wireframe_cube_local((0, 1, 0))
             elif type == "CUBE":
                 self.draw_solid_cube_local(rgba)
             elif type == "PYRAMID":
                 self.draw_solid_pyramid_local(rgba)
             elif type == "SPHERE":
                  glColor4f(*rgba)
                  gluSphere(self.quadric, 0.5, segments, segments)

        else:
            # Selection Visualization: Semi-transparent + Moving Dotted Outline
            if is_selected:
                 r,g,b = color if len(color)==3 else color[:3]
                 draw_color = (r, g, b, 0.7) # Transparent
            else:
                 draw_color = color
            
            # 1. Draw Solid
            # Enable Blending if selected (transparency)
            if is_selected:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(GL_FALSE)

            if type == "CUBE":
                 self.draw_solid_cube_local(draw_color)
            elif type == "PYRAMID":
                 self.draw_solid_pyramid_local(draw_color)
            elif type == "SPHERE":
                 if len(draw_color) == 4: glColor4f(*draw_color)
                 else: glColor3f(*draw_color)
                 gluSphere(self.quadric, 0.5, segments, segments)
            
            if is_selected:
                glDepthMask(GL_TRUE)
                glDisable(GL_BLEND)
                
                # 2. Draw Moving Dotted Outline
                glEnable(GL_LINE_STIPPLE)
                
                # Animate stipple: Shift pattern based on time
                # We need a time variable. passing in time.time() would be best, 
                # but for now we can just use a static counter from pygame or time module
                import time
                shift = int(time.time() * 10) % 16
                glLineStipple(2, 0xAAAA) # Simple dotted/dashed pattern
                
                # Using a slightly larger wireframe to prevent z-fighting
                glPushMatrix()
                glScalef(1.01, 1.01, 1.01)
                
                if type == "CUBE":
                    self.draw_wireframe_cube_local((1,1,1))
                elif type == "PYRAMID":
                    self.draw_wireframe_pyramid_local((1,1,1))
                elif type == "SPHERE":
                    # Sphere wireframe is tricky with glut/glu, just draw a box or rings?
                    # Let's draw the bounding box for selection mostly
                    self.draw_wireframe_cube_local((1,1,1))
                    
                glPopMatrix()
                glDisable(GL_LINE_STIPPLE)
        
        glPopMatrix()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
        return True

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0, 0, self.camera_z)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glTranslatef(-self.target_x, -self.target_y, -self.target_z)
        
        if self.cursor_pos:
            glLightfv(GL_LIGHT1, GL_POSITION, [*self.cursor_pos, 1.0])
        else:
            glLightfv(GL_LIGHT1, GL_POSITION, [0, 0, 0, 1])

        # --- DRAW WORLD ---
        glPushMatrix()
        
        # 1. Shapes
        for i, shape in enumerate(self.shapes):
            if "center" in shape:
                is_sel = (i in self.selected_shape_indices)
                self.render_shape(shape, is_ghost=False, is_selected=is_sel)
        
        # 2. Preview Shape
        if hasattr(self, 'current_preview_shape') and self.current_preview_shape:
             glEnable(GL_BLEND)
             glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
             self.render_shape(self.current_preview_shape, is_ghost=True)
             glDisable(GL_BLEND)
             self.current_preview_shape = None

        # 3. Strokes
        for i, stroke_data in enumerate(self.lines):
            stroke = stroke_data["points"]
            
            # Unpack color
            base_color = stroke_data.get("color", (1,1,1))
            if len(base_color) == 4: r,g,b,a = base_color
            else: r,g,b = base_color; a = 1.0

            thickness = stroke_data.get("thickness", 1)
            radius = 0.01 + (thickness * 0.005)
            
            is_selected = (i in self.selected_indices)
            
            if is_selected:
                 glEnable(GL_BLEND)
                 glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                 color = (r, g, b, 0.7) # Transparent
            else:
                 color = (r, g, b, a)

            if len(stroke) > 1:
                for j in range(len(stroke) - 1):
                    self.draw_cylinder(stroke[j], stroke[j+1], radius, color)
            
            if is_selected:
                 glDisable(GL_BLEND)
                 
                 # Draw View-Dependent Silhouette Lines
                 glDisable(GL_DEPTH_TEST) # Ensure visibility
                 glEnable(GL_LINE_STIPPLE)
                 import time 
                 shift = int(time.time() * 10) % 16
                 glLineStipple(2, 0xAAAA) 
                 
                 glLineWidth(2.0)
                 glColor3f(1.0, 1.0, 1.0) # White outline
                 
                 # Calculate Silhouette Points
                 cam_pos = self.get_camera_position_world()
                 pts = [np.array(p) for p in stroke]
                 
                 left_line = []
                 right_line = []
                 
                 # We need at least 2 points to define a tangent
                 if len(pts) >= 2:
                     for k in range(len(pts)):
                         p = pts[k]
                         
                         # Calculate Tangent T
                         if k == 0:
                             t = pts[k+1] - p
                         elif k == len(pts) - 1:
                             t = p - pts[k-1]
                         else:
                             # Average tangent
                             t = pts[k+1] - pts[k-1]
                         
                         t_len = np.linalg.norm(t)
                         if t_len == 0: continue
                         t = t / t_len
                         
                         # Calculate View Vector V
                         v = cam_pos - p
                         v_len = np.linalg.norm(v)
                         if v_len == 0: continue
                         v = v / v_len
                         
                         # Calculate Side Vector S = Cross(V, T)
                         s = np.cross(v, t)
                         s_len = np.linalg.norm(s)
                         # If looking straight down the tube, s might be 0
                         if s_len < 0.001: 
                             # Fallback? just skip
                             continue
                         s = s / s_len
                         
                         # Offset points
                         offset = s * radius * 1.0 # Exact radius or slightly larger?
                         left_line.append(p + offset)
                         right_line.append(p - offset)
                 
                 # Draw Left Line
                 glBegin(GL_LINE_STRIP)
                 for p in left_line:
                     glVertex3f(*p)
                 glEnd()
                 
                 # Draw Right Line
                 glBegin(GL_LINE_STRIP)
                 for p in right_line:
                     glVertex3f(*p)
                 glEnd()
                 
                 glLineWidth(1.0)
                 glDisable(GL_LINE_STIPPLE)
                 glEnable(GL_DEPTH_TEST)

        # 4. Current Stroke
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        r, g, b = self.current_color
        glColor4f(r, g, b, 0.5)
        curr_radius = 0.01 + (self.current_thickness * 0.005)
        if len(self.current_stroke) > 1:
            for i in range(len(self.current_stroke) - 1):
                 self.draw_cylinder(self.current_stroke[i], self.current_stroke[i+1], curr_radius, (r,g,b))
        if self.current_stroke:
             p = self.current_stroke[-1]
             glPushMatrix()
             glTranslatef(p[0], p[1], p[2])
             gluSphere(self.quadric, curr_radius, 8, 8)
             glPopMatrix()
        glDisable(GL_BLEND)

        # 5. Cursor
        if self.cursor_pos:
            glPushMatrix()
            glTranslatef(*self.cursor_pos)
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 0.0, 1.0)
            gluSphere(self.quadric, 0.15, 10, 10)
            glEnable(GL_LIGHTING)
            glPopMatrix()

        glPopMatrix() 
        pygame.display.flip()

    def draw_cylinder(self, p1, p2, radius, color):
        dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist == 0: return

        glPushMatrix()
        glTranslatef(p1[0], p1[1], p1[2])
        if len(color) == 4:
            glColor4f(*color)
        else:
            glColor3f(*color)
        
        v = np.array([dx, dy, dz]) / dist
        z_axis = np.array([0, 0, 1])
        r_axis = np.cross(z_axis, v)
        
        if np.linalg.norm(r_axis) == 0:
            if dz < 0: glRotatef(180, 1, 0, 0)
        else:
            angle = math.degrees(math.acos(np.clip(np.dot(z_axis, v), -1.0, 1.0)))
            glRotatef(angle, *r_axis)
        
        gluCylinder(self.quadric, radius, radius, dist, 8, 1)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(p1[0], p1[1], p1[2])
        gluSphere(self.quadric, radius, 8, 8)
        glPopMatrix()
