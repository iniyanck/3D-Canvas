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
        self.points = [] # List of (x, y, z) tuples
        self.lines = []  # List of lists of points (strokes)
        self.shapes = [] # List of shape dicts: {type, pos, scale, color}
        self.current_stroke = []
        self.cursor_pos = None # (x, y, z) tuple for 3D cursor
        
        # Tool State
        self.selected_indices = [] # Indices of selected strokes in self.lines
        self.undo_stack = []
        self.redo_stack = []
        
        # Rotation State
        self.rot_x = 0.0 # Pitch
        self.rot_y = 0.0 # Yaw
        
        # Camera State
        # Camera State
        self.camera_z = -3.0 # Distance from pivot (Closer = Pivot feels more "centered" on view)
        self.target_x = 0.0 # Pivot Point X
        self.target_y = 0.0 # Pivot Point Y

        # Quadric for cylinders/spheres
        self.quadric = gluNewQuadric()


        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Gesture Canvas")

        # Basic OpenGL Setup
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, self.camera_z) # Move camera back
        
        # Lighting Setup
        glEnable(GL_DEPTH_TEST) # Enable depth testing for 3D solidity
        glEnable(GL_LIGHTING)   # Enable Lighting
        glEnable(GL_COLOR_MATERIAL) # Enable material tracking of color
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light 0: Static Scene Light (Directional)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 10, 10, 0]) # Directional light from top-right-front
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Light 1: Dynamic Cursor Light (Point)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 1.0, 0.8, 1]) # Warm light
        glLightfv(GL_LIGHT1, GL_SPECULAR, [1.0, 1.0, 1.0, 1])
        glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.2)
        glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.02)

    def save_state(self):
        # Deep copy of lines for undo
        import copy
        self.undo_stack.append(copy.deepcopy(self.lines))
        # Limit stack size
        if len(self.undo_stack) > 20:
             self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.lines)
            self.lines = self.undo_stack.pop()
            self.selected_indices = [] # Clear selection on undo

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.lines)
            self.lines = self.redo_stack.pop()
            self.selected_indices = []

    def get_world_point_from_view(self, x, y, z_relative):
        """
        Converts screen (x, y) and View-Relative Depth (z_relative) to 3D World Coordinates.
        
        This makes the interaction 'View Consistent'. Moving your hand RIGHT will always move
        the cursor RIGHT on the screen, regardless of how the 3D world is rotated.
        
        z_relative: Depth relative to the camera/screen plane. 
                    0 = Standard Drawing Plane.
                    Positive = Closer to Camera.
                    Negative = Further from Camera.
        """
        # 1. Normalized Device Coordinates
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1)
        
        # 2. View Space Point
        fov = 45
        aspect = self.width / self.height
        tan_half_fov = math.tan(math.radians(fov / 2))
        
        # Base distance for the drawing plane (In front of camera)
        # We placed camera at self.camera_z (negative value). 
        # So a point at World(0,0,0) is at Distance = -self.camera_z from Camera.
        # Let's say standard drawing depth is fixed distance from Camera.
        DRAWING_DIST = -self.camera_z # Draw roughly at the pivot point
        
        # Apply hand depth (z_relative is roughly -2 to +2 now)
        # Closer Hand (Positive Z) -> Smaller Depth (Closer to Camera)
        # Farther Hand (Negative Z) -> Larger Depth (Farther from Camera)
        depth = DRAWING_DIST - z_relative * 1.0 # Tuned to 1.0 from 25.0
        
        # Clamp depth to avoid behind camera clipping (Camera at 0, Near Plane at 0.1)
        depth = max(0.5, depth)
        
        # View Space Coordinates
        # x_view = x_ndc * depth * aspect * tan_half_fov
        # y_view = y_ndc * depth * tan_half_fov
        # z_view = -depth
        
        x_view = norm_x * depth * aspect * tan_half_fov
        y_view = norm_y * depth * tan_half_fov
        z_view = -depth
        
        p_view = np.array([x_view, y_view, z_view])
        
        # 3. Transform View Space -> World Space
        # New Orbit Logic: View = T_back * R * T_center_inv * World
        # World = T_center * InvR * InvT_back * View
        
        # InvT_back (0, 0, camera_z)
        p_temp = p_view - np.array([0, 0, self.camera_z])
        
        # InvR (Same as before)
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
        
        # InvT_center_inv = T_center (+target_x, +target_y)
        p_world = p_temp + np.array([self.target_x, self.target_y, 0])
        
        return tuple(p_world)

    def get_interface_position(self, x, y, z):
        """
        Returns the screen pixel coordinates (sx, sy) where the 3D cursor 
        corresponding to input screen(x, y) and relative_z would appear.
        This is basically identity because input x,y ARE screen coords.
        But we might want to know where the *projected world point* is?
        If we use view-dependent mapping, input (x,y) -> World Point -> Projects back to same (x,y).
        So just return x,y.
        """
        return x, y

    def add_point(self, x, y, z, start_new_stroke=False):
        world_point = self.get_world_point_from_view(x, y, z)

        if start_new_stroke:
            if self.current_stroke:
                self.save_state() # Save before adding new stroke
                self.lines.append(self.current_stroke)
                self.current_stroke = []
                
        # Spatial Filtering: Don't add if too close to last point
        MIN_DIST = 0.02 # Tune this for smoothness vs detail
        if self.current_stroke:
            last_p = np.array(self.current_stroke[-1])
            new_p = np.array(world_point)
            dist = np.linalg.norm(new_p - last_p)
            if dist < MIN_DIST:
                return # Skip this point
        
        self.current_stroke.append(world_point)

    def end_stroke(self):
        """
        Force the current stroke to be committed to the list of fixed lines.
        Useful for when the user stops drawing.
        """
        if self.current_stroke:
            self.save_state() # Save before committing
            self.lines.append(self.current_stroke)
            self.current_stroke = []

    def update_cursor(self, x, y, z):
        """
        Update the position of the 3D cursor (without drawing).
        """
        # Use precise unprojection
        self.cursor_pos = self.get_world_point_from_view(x, y, z)


    def clear(self):
        self.save_state()
        self.lines = []
        self.current_stroke = []
        self.selected_indices = []

    def rotate(self, d_yaw, d_pitch):
        """
        Accumulate rotation angles.
        d_yaw: Change in Y rotation (horizontal movement)
        d_pitch: Change in X rotation (vertical movement)
        """
        self.rot_y += d_yaw
        self.rot_x += d_pitch

    def rotate_selection(self, d_yaw, d_pitch):
        """
        Rotate only the selected lines.
        This is a geometry transformation on the points themselves.
        """
        if not self.selected_indices:
             return
        
        # We need to rotate the points of the selected strokes around the centroid of the selection?
        # Or just around the world origin? Rotation usually around origin or local origin.
        # User asked: "Or if you just do the rotate/pan while selected, it moves around/rotates the selected shape"
        # Let's rotate around the Centroid of the selection for intuitive manipulation.
        
        # 1. Calc Centroid
        all_points = []
        for idx in self.selected_indices:
             all_points.extend(self.lines[idx])
        
        if not all_points: return
        
        centroid = np.mean(all_points, axis=0)
        
        # 2. Rotate points
        rad_x = np.radians(d_pitch)
        rad_y = np.radians(d_yaw)
        
        # Prepare rotation matrices
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
        
        # Combined Rotation R = Ry * Rx (Order matters, let's just do one by one)
        
        # Ideally we want to rotate around Camera Right and Camera Up vectors if we want view-relative rotation...
        # But `d_yaw` and `d_pitch` from fist are just delta X and delta Y movements mapped to rotation angles.
        
        # Let's just rotate around world axes for now.
        
        new_lines = []
        import copy
        # We should modify self.lines directly? 
        # But we need to save state first if this is a discrete action?
        # Continuous rotation should probably not save state every frame.
        # Save state on Interaction Start (in main.py) would be better.
        
        for idx in self.selected_indices:
            stroke = self.lines[idx]
            new_stroke = []
            for p in stroke:
                v = np.array(p) - centroid
                # Rotate Y
                v = np.dot(ry_mat, v)
                # Rotate X
                v = np.dot(rx_mat, v)
                new_p = v + centroid
                new_stroke.append(tuple(new_p))
            self.lines[idx] = new_stroke

    def move_selection(self, dx, dy, dz=0.0):
        """
        Move selected lines in screen plane (approximate).
        """
        scale_xy = 0.01
        scale_z = 0.5 # Boost Z sensitivity
        
        # Hand Right (+dx) -> Object Right (+X).
        # Hand Pull (Back/-dz) -> Object Closer (+Z). (So -dz)
        move_vec = np.array([dx * scale_xy, -dy * scale_xy, -dz * scale_z])
        
        # We should rotate this move_vec by the inverse of the Camera Rotation
        # so that "Right" means "Right on Screen".
        rad_x = np.radians(-self.rot_x)
        rad_y = np.radians(-self.rot_y)
        
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
        
        # vec_world = InvRotY * InvRotX * vec_view
        move_vec = np.dot(rx_mat, move_vec)
        move_vec = np.dot(ry_mat, move_vec)
        
        for idx in self.selected_indices:
            stroke = self.lines[idx]
            new_stroke = []
            for p in stroke:
                new_p = np.array(p) + move_vec
                new_stroke.append(tuple(new_p))
            self.lines[idx] = new_stroke

    def pan_camera(self, dx, dy, dz=0.0):
        """
        Pan by moving the Pivot Target.
        """
        scale_xy = 0.02
        scale_z = 1.0 # Boost Z sensitivity for Zoom
        
        # Hand Right (dx>0) -> Target Left (-dx) -> Content Right. Correct.
        # Hand Up (dy<0) -> Target Down (+dy neg) -> Content Up. Correct.
        # Pull (dz<0) -> Camera Z Increase (Closer). Correct.
        
        pan_vec = np.array([-dx * scale_xy, dy * scale_xy, 0])
        
        rad_x = np.radians(-self.rot_x)
        rad_y = np.radians(-self.rot_y)
        
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
        
        pan_vec = np.dot(rx_mat, pan_vec)
        pan_vec = np.dot(ry_mat, pan_vec)
        
        self.target_x += pan_vec[0]
        self.target_y += pan_vec[1]
        
        # Apply Depth (dz) to Camera Z (Zoom)
        # Invert dz: Pull (neg) -> Zoom In (Increase Cam Z)
        self.camera_z -= dz * scale_z
        self.camera_z = max(-20.0, min(-0.5, self.camera_z))


    def select_at(self, screen_x, screen_y, radius=30):
        """
        Finds the stroke closest to the screen coordinates.
        Toggles selection if found.
        """
        # First, deselect all if clicked on empty space?
        # Or add to selection?
        # Let's simple toggle: Click -> Select. Click Empty -> Deselect All.
        
        cursor_pos = np.array([screen_x, screen_y])
        best_dist = radius
        best_idx = -1
        
        for i, stroke in enumerate(self.lines):
            # Check stroke points
            for p3d in stroke:
                p2d = self.project_point(p3d)
                if p2d:
                    dist = np.linalg.norm(np.array(p2d) - cursor_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
        
        if best_idx != -1:
            if best_idx in self.selected_indices:
                self.selected_indices.remove(best_idx)
            else:
                self.selected_indices = [best_idx] # Single selection for now? Or additive? 
                # User asked for "Select logic". Let's do Single Select + Click Empty to Clear.
        else:
            self.selected_indices = []

    def zoom(self, d_z):
        """
        Adjust camera Z position (Zoom).
        d_z: Change in Z depth. Positive moves camera forward (Zoom In), Negative moves back (Zoom Out).
        """
        # Sensitivity
        self.camera_z += d_z * 2.0 
        
        # Clamp zoom to reasonable limits
        # -2.0 is very close (objects at 0 are 2 units away)
        # -20.0 is far
        self.camera_z = max(-20.0, min(-2.0, self.camera_z))

    def get_distance_to_segment(self, p1, p2, point):
        p1 = np.array(p1)
        p2 = np.array(p2)
        point = np.array(point)
        
        ab = p2 - p1
        if np.linalg.norm(ab) == 0:
            return np.linalg.norm(point - p1)
            
        ap = point - p1
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + t * ab
        return np.linalg.norm(point - closest)

    def project_point(self, point_world):
        """
        Projects a 3D world point to 2D screen coordinates.
        Returns (screen_x, screen_y) or None if behind camera.
        """
        x, y, z = point_world
        
        # 1. Apply Rotation (ModelView Rotation)
        # Note: In render(), we do glRotatef(self.rot_x, 1, 0, 0) then glRotatef(self.rot_y, 0, 1, 0)
        # So we must apply Y rotation then X rotation to the point? 
        # Wait, standard OpenGL Fixed Function:
        # glRotate(x) -> Multiplies CurrentMatrix * RotX
        # glRotate(y) -> Multiplies CurrentMatrix * RotY
        # Vertex = Matrix * v
        # So Vertex = (RotX * RotY) * v -> First Y then X is applied to vector?
        # Let's verify standard order. 
        # If I call rotX then rotY, the matrix is M = RotX * RotY.
        # P_eye = M * P_world = RotX * (RotY * P_world).
        # So yes, Yaw (Y) first, then Pitch (X).
        
        # 0. Apply Target Offset (Important Fix!)
        # Render Order: T(cz) * Rot * T(-target)
        # Projection Order: Rot * (P - Target) + cz ??
        # Yes: V_view = Rot * (P_world - Target) + [0,0,cz]
        
        tx, ty, tz = x - self.target_x, y - self.target_y, z
        
        rad_x = np.radians(self.rot_x)
        rad_y = np.radians(self.rot_y)
        
        # Rotate Y
        rx, ry, rz = tx, ty, tz
        new_x = rx * np.cos(rad_y) + rz * np.sin(rad_y)
        new_z = -rx * np.sin(rad_y) + rz * np.cos(rad_y)
        rx, rz = new_x, new_z
        
        # Rotate X
        new_y = ry * np.cos(rad_x) - rz * np.sin(rad_x)
        new_z = ry * np.sin(rad_x) + rz * np.cos(rad_x)
        ry, rz = new_y, new_z
        
        # 2. Apply Translation (Camera Move)
        # We did glTranslatef(0, 0, self.camera_z)
        rz += self.camera_z
        
        # 3. Perspective Projection
        if rz >= -0.1: # Near clipping plane roughly
            return None
            
        # FOV 45 degrees.
        # tan(fov/2) = (h/2) / d
        # projected_y = y / z ?
        
        # Let's use standard projection formulas or gluProject if we could, 
        # but manual is fine for simple setup.
        fov = 45
        aspect = self.width / self.height
        f = 1.0 / math.tan(math.radians(fov) / 2)
        
        # Normalized Device Coordinates
        # x_ndc = (f / aspect) * x / -z
        # y_ndc = f * y / -z
        
        x_ndc = (f / aspect) * rx / -rz
        y_ndc = f * ry / -rz
        
        # Screen Coordinates
        # x_screen = (x_ndc + 1) * width / 2
        # y_screen = (1 - y_ndc) * height / 2  (Flip Y for screen coords)
        
        screen_x = (x_ndc + 1) * self.width / 2
        screen_y = (1 - y_ndc) * self.height / 2
        
        return screen_x, screen_y

    def get_distance_to_segment_2d(self, p1, p2, point):
        p1 = np.array(p1)
        p2 = np.array(p2)
        point = np.array(point)
        
        ab = p2 - p1
        if np.linalg.norm(ab) == 0:
            return np.linalg.norm(point - p1)
            
        ap = point - p1
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + t * ab
        return np.linalg.norm(point - closest)

    def erase_at(self, screen_x, screen_y, z=None, radius=30):
        """
        Erase strokes that overlap with the 2D screen circle defined by screen_x, screen_y, radius.
        radius is in screen pixels.
        """
        self.save_state() # Save state before erasing
        
        eraser_pos = np.array([screen_x, screen_y])
        new_lines = []
        
        # Also check current stroke
        if self.current_stroke:
             self.lines.append(self.current_stroke)
             self.current_stroke = []

        for i, stroke in enumerate(self.lines):
            # If selected, maybe don't erase? Or erase freely. Erase free is standard.
            
            new_stroke = []
            current_segment = []
            
            # Optimization: Check bounding box of stroke first (screen space)? 
            # Skip for now.
            
            # Segment based erasure (Simpler version: Delete checks points)
            keep_stroke = True
            
            for p3d in stroke:
                p2d = self.project_point(p3d)
                if p2d is None:
                    current_segment.append(p3d)
                    continue
                
                dist = np.linalg.norm(np.array(p2d) - eraser_pos)
                if dist <= radius:
                    # Break segment
                    if current_segment:
                        new_lines.append(current_segment)
                    current_segment = []
                else:
                    current_segment.append(p3d)
            
            if current_segment:
                new_lines.append(current_segment)
                
        self.lines = new_lines
        
        # Erase Shapes too
        new_shapes = []
        for shape in self.shapes:
             # Check if bounding box overlaps roughly
             keep = True
             if "bounds" in shape:
                 p1, p2 = shape["bounds"]
                 # Simple check: Is center close? Or project corners
                 center = (np.array(p1)+np.array(p2)) / 2
                 p2d = self.project_point(center)
                 if p2d:
                     dist = np.linalg.norm(np.array(p2d) - eraser_pos)
                     # Approximate radius for shape:
                     shape_r = np.linalg.norm(np.array(p2) - np.array(p1)) * 0.5
                     # Project shape_r to screen? Hard.
                     # Simple heuristic: if center is within eraser radius + 20 px
                     if dist < radius + 10: 
                         keep = False
             if keep:
                 new_shapes.append(shape)
        self.shapes = new_shapes
        
        self.selected_indices = [] # Clear selection after erase to avoid index errors

    def add_shape(self, shape_type, x, y, z, scale=0.5):
        """
        Add a 3D shape at the given position.
        Shapes are just strokes (lines) formed into the shape.
        """
        center = self.get_world_point_from_view(x, y, z)
        cx, cy, cz = center
        
        strokes = []
        s = scale
        
        if shape_type == "CUBE":
            # 12 Edges
            # Bottom Square being z-s/2
            corners = [
                (cx-s, cy-s, cz-s), (cx+s, cy-s, cz-s), (cx+s, cy+s, cz-s), (cx-s, cy+s, cz-s),
                (cx-s, cy-s, cz+s), (cx+s, cy-s, cz+s), (cx+s, cy+s, cz+s), (cx-s, cy+s, cz+s)
            ]
            # Bottom Loop
            strokes.append([corners[0], corners[1], corners[2], corners[3], corners[0]])
            # Top Loop
            strokes.append([corners[4], corners[5], corners[6], corners[7], corners[4]])
            # Vertical Pillars
            strokes.append([corners[0], corners[4]])
            strokes.append([corners[1], corners[5]])
            strokes.append([corners[2], corners[6]])
            strokes.append([corners[3], corners[7]])
            
        elif shape_type == "SPHERE":
            # Lat/Lon lines
            steps = 10
            for i in range(steps):
                lat = math.pi * (i / steps)
                # ... Math for sphere rings ...
                pass
            # Just add a simple cross for now to test
            strokes.append([(cx-s, cy, cz), (cx+s, cy, cz)])
            strokes.append([(cx, cy-s, cz), (cx, cy+s, cz)])
            strokes.append([(cx, cy, cz-s), (cx, cy, cz+s)])

        elif shape_type == "PYRAMID":
            top = (cx, cy, cz+s)
            base = [
                (cx-s, cy-s, cz-s), (cx+s, cy-s, cz-s), (cx+s, cy+s, cz-s), (cx-s, cy+s, cz-s)
            ]
            # Base
            strokes.append([base[0], base[1], base[2], base[3], base[0]])
            # Sides
            for b in base:
                strokes.append([b, top])
        
        self.lines.extend(strokes)

        self.lines.extend(strokes)

    def add_shape_bounds(self, p1, p2, shape_type, color=(1.0, 1.0, 1.0)):
        """
        Add a solid shape defined by two corner points (min/max).
        """
        # Calculate bounds
        pos1 = np.array(p1)
        pos2 = np.array(p2)
        
        self.shapes.append({
            "type": shape_type,
            "bounds": (pos1, pos2),
            "color": color
        })

    def draw_solid_cube_bounds(self, p1, p2, color):
        min_x = min(p1[0], p2[0])
        max_x = max(p1[0], p2[0])
        min_y = min(p1[1], p2[1])
        max_y = max(p1[1], p2[1])
        min_z = min(p1[2], p2[2])
        max_z = max(p1[2], p2[2])
        
        glPushMatrix()
        # No translate, we draw in world coords directly
        glColor3f(*color)
        
        glBegin(GL_QUADS)
        
        # Front Face (Z Max)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(min_x, min_y, max_z)
        glVertex3f(max_x, min_y, max_z)
        glVertex3f(max_x, max_y, max_z)
        glVertex3f(min_x, max_y, max_z)
        
        # Back Face (Z Min)
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(min_x, min_y, min_z)
        glVertex3f(min_x, max_y, min_z)
        glVertex3f(max_x, max_y, min_z)
        glVertex3f(max_x, min_y, min_z)
        
        # Top Face (Y Max)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(min_x, max_y, min_z)
        glVertex3f(min_x, max_y, max_z)
        glVertex3f(max_x, max_y, max_z)
        glVertex3f(max_x, max_y, min_z)
        
        # Bottom Face (Y Min)
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(min_x, min_y, min_z)
        glVertex3f(max_x, min_y, min_z)
        glVertex3f(max_x, min_y, max_z)
        glVertex3f(min_x, min_y, max_z)
        
        # Right face (X Max)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(max_x, min_y, min_z)
        glVertex3f(max_x, max_y, min_z)
        glVertex3f(max_x, max_y, max_z)
        glVertex3f(max_x, min_y, max_z)
        
        # Left Face (X Min)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(min_x, min_y, min_z)
        glVertex3f(min_x, min_y, max_z)
        glVertex3f(min_x, max_y, max_z)
        glVertex3f(min_x, max_y, min_z)
        
        glEnd()
        glPopMatrix()

    def draw_wireframe_cube_bounds(self, p1, p2, color):
        min_x = min(p1[0], p2[0])
        max_x = max(p1[0], p2[0])
        min_y = min(p1[1], p2[1])
        max_y = max(p1[1], p2[1])
        min_z = min(p1[2], p2[2])
        max_z = max(p1[2], p2[2])
        
        glPushMatrix()
        glColor3f(*color)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(2.0)
        
        glBegin(GL_QUADS)
        # Front
        glVertex3f(min_x, min_y, max_z); glVertex3f(max_x, min_y, max_z); glVertex3f(max_x, max_y, max_z); glVertex3f(min_x, max_y, max_z)
        # Back
        glVertex3f(min_x, min_y, min_z); glVertex3f(max_x, min_y, min_z); glVertex3f(max_x, max_y, min_z); glVertex3f(min_x, max_y, min_z)
        # Top
        glVertex3f(min_x, max_y, min_z); glVertex3f(min_x, max_y, max_z); glVertex3f(max_x, max_y, max_z); glVertex3f(max_x, max_y, min_z)
        # Bottom
        glVertex3f(min_x, min_y, min_z); glVertex3f(max_x, min_y, min_z); glVertex3f(max_x, min_y, max_z); glVertex3f(min_x, min_y, max_z)
        # Right
        glVertex3f(max_x, min_y, min_z); glVertex3f(max_x, max_y, min_z); glVertex3f(max_x, max_y, max_z); glVertex3f(max_x, min_y, max_z)
        # Left
        glVertex3f(min_x, min_y, min_z); glVertex3f(min_x, min_y, max_z); glVertex3f(min_x, max_y, max_z); glVertex3f(min_x, max_y, min_z)
        glEnd()
        
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()

    def add_solid_shape(self, shape_type, x, y, z, scale=0.5, color=(0.0, 0.0, 1.0)):
        # Deprecated logic kept for compatibility if needed, but we prefer bounds now.
        # Convert pt+scale to bounds
        center = self.get_world_point_from_view(x, y, z)
        p1 = center - scale
        p2 = center + scale
        # White color default
        self.add_shape_bounds(p1, p2, shape_type, color=(1.0, 1.0, 1.0))
        
    def preview_shape_bounds(self, p1, p2, shape_type):
        self.current_preview_shape = {"type": shape_type, "bounds": (p1, p2)}

    def draw_solid_cube(self, center, size, color):
         # Legacy wrapper
         self.draw_solid_cube_bounds(np.array(center)-size, np.array(center)+size, color)

    def draw_solid_sphere(self, center, radius, color):
        cx, cy, cz = center
        glPushMatrix()
        glTranslatef(cx, cy, cz)
        glColor3f(*color)
        gluSphere(self.quadric, radius, 16, 16)
        glPopMatrix()

    def draw_solid_pyramid(self, center, size, color):
        # Todo: bounds version
        cx, cy, cz = center
        s = size
        glPushMatrix()
        glTranslatef(cx, cy, cz)
        glColor3f(*color)
        
        glBegin(GL_TRIANGLES)
        # Front
        glNormal3f(0.0, 0.5, 1.0) # Approx normal
        glVertex3f( 0.0, s, 0.0)
        glVertex3f(-s, -s, s)
        glVertex3f( s, -s, s)
        
        # Right
        glNormal3f(1.0, 0.5, 0.0)
        glVertex3f(0.0, s, 0.0)
        glVertex3f(s, -s, s)
        glVertex3f(s, -s, -s)
        
        # Back
        glNormal3f(0.0, 0.5, -1.0)
        glVertex3f(0.0, s, 0.0)
        glVertex3f(s, -s, -s)
        glVertex3f(-s, -s, -s)
        
        # Left
        glNormal3f(-1.0, 0.5, 0.0)
        glVertex3f(0.0, s, 0.0)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glEnd()
        
        # Bottom
        glBegin(GL_QUADS)
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, -s, -s)
        glVertex3f( s, -s, -s)
        glVertex3f( s, -s, s)
        glEnd()

        glPopMatrix()

    def preview_shape(self, shape_type, x, y, z, scale=0.5):
        # Legacy
        pass # We use bounds now

    def draw_cylinder(self, p1, p2, radius=0.03):
        """
        Draws a cylinder between two points p1 and p2.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist == 0:
            return

        glPushMatrix()
        glTranslatef(p1[0], p1[1], p1[2])
        
        # Rotating the cylinder to align with the vector p1->p2
        # Default GluCylinder is along Z axis.
        v = np.array([dx, dy, dz]) / dist
        z_axis = np.array([0, 0, 1])
        
        # Axis of rotation = z_axis x v
        r_axis = np.cross(z_axis, v)
        
        # Check for zero vector (parallel)
        if np.linalg.norm(r_axis) == 0:
             # Angle is 0 or 180.
             if dz < 0: # 180 degrees
                 glRotatef(180, 1, 0, 0)
             # else 0 degrees, do nothing
        else:
             # Angle = acos(z_axis . v)
             dot_product = np.dot(z_axis, v)
             angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
             glRotatef(angle, *r_axis)
        
        gluCylinder(self.quadric, radius, radius, dist, 16, 1) # Increased slices 6 -> 16
        glPopMatrix()
        
        # Draw spheres at joints for smoothness
        glPushMatrix()
        glTranslatef(p1[0], p1[1], p1[2])
        gluSphere(self.quadric, radius, 16, 16) # Increased slices 6 -> 16
        glPopMatrix()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity() # Reset view matrix
        
        # Camera Setup
        # Camera Setup
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        # Orbit Transform
        glTranslatef(0, 0, self.camera_z) # Move back
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glTranslatef(-self.target_x, -self.target_y, 0) # Center on target
        
        # Update Dynamic Light (Cursor Light) Position
        if self.cursor_pos:
            # The light position is defined in eye coordinates if called after view transform
            pos_x, pos_y, pos_z = self.cursor_pos
            # We want the light to be AT the cursor position relative to the camera
            glLightfv(GL_LIGHT1, GL_POSITION, [pos_x, pos_y, pos_z, 1.0]) # 1.0 w = positional light
        else:
             glLightfv(GL_LIGHT1, GL_POSITION, [0, 0, 0, 1])

        # --- World Rendering (Rotated) ---
        glPushMatrix()
        # Rotation already applied in view, so we are in world coords!
        # DO NOT APPLY ROTATION HERE AGAIN!
        # Wait, previous logic was: Camera Translate -> Rotate World.
        # Orbit Logic: Camera View Matrix includes Rotation.
        # So "Coordinate System" is World Space from here on.
        # So remove glRotatef here.
        
        # Draw Solid Shapes
        for shape in self.shapes:
            if "bounds" in shape:
                if shape["type"] == "CUBE":
                    self.draw_solid_cube_bounds(shape["bounds"][0], shape["bounds"][1], shape["color"])
                # Fallback for others (TODO)
                elif shape["type"] == "SPHERE":
                     p1, p2 = shape["bounds"]
                     # bounds to center + scale
                     center = (p1 + p2) / 2
                     dims = np.abs(p2 - p1) / 2 # Half-extents
                     
                     glPushMatrix()
                     glTranslatef(*center)
                     glScalef(*dims)
                     glColor3f(*shape["color"])
                     # Draw Unit Sphere
                     gluSphere(self.quadric, 1.0, 16, 16)
                     glPopMatrix()

                elif shape["type"] == "PYRAMID":
                     p1, p2 = shape["bounds"]
                     center = (p1 + p2) / 2
                     dims = np.abs(p2 - p1) / 2
                     
                     glPushMatrix()
                     glTranslatef(*center)
                     glScalef(*dims)
                     glColor3f(*shape["color"])
                     # Draw Unit Pyramid (Base at -1 to 1, Tip at +1??)
                     # My draw_solid_pyramid was centered at 0, size s.
                     # Let's inline a unit pyramid here or call a helper.
                     # Unit Pyramid: Base on Y=-1 plane? Tip at Y=1?
                     # Let's define a Unit Pyramid fitting in [-1,1]^3
                     
                     glBegin(GL_TRIANGLES)
                     # Front (Z+)
                     glNormal3f(0, 0.5, 1); glVertex3f(0, 1, 0); glVertex3f(-1, -1, 1); glVertex3f(1, -1, 1)
                     # Right (X+)
                     glNormal3f(1, 0.5, 0); glVertex3f(0, 1, 0); glVertex3f(1, -1, 1); glVertex3f(1, -1, -1)
                     # Back (Z-)
                     glNormal3f(0, 0.5, -1); glVertex3f(0, 1, 0); glVertex3f(1, -1, -1); glVertex3f(-1, -1, -1)
                     # Left (X-)
                     glNormal3f(-1, 0.5, 0); glVertex3f(0, 1, 0); glVertex3f(-1, -1, -1); glVertex3f(-1, -1, 1)
                     glEnd()
                     
                     # Base
                     glBegin(GL_QUADS)
                     glNormal3f(0, -1, 0)
                     glVertex3f(-1, -1, 1); glVertex3f(-1, -1, -1); glVertex3f(1, -1, -1); glVertex3f(1, -1, 1)
                     glEnd()

                     glPopMatrix()
            else:
                 # Legacy
                 if shape["type"] == "CUBE":
                     self.draw_solid_cube(shape["pos"], shape["scale"], shape["color"])
                 elif shape["type"] == "SPHERE":
                     self.draw_solid_sphere(shape["pos"], shape["scale"], shape["color"])
                 elif shape["type"] == "PYRAMID":
                     self.draw_solid_pyramid(shape["pos"], shape["scale"], shape["color"])
        
        # Draw Preview Shape if exists
        if hasattr(self, 'current_preview_shape') and self.current_preview_shape:
             s = self.current_preview_shape
             glEnable(GL_BLEND)
             glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
             # Semi-transparent ghost
             
             if "bounds" in s:
             # Semi transparent inner shape
                 p1, p2 = s["bounds"]
                 type = s.get("type", "CUBE")
                 
                 # Draw WIREFRAME Bounds
                 self.draw_wireframe_cube_bounds(p1, p2, (1, 1, 1))

                 # Draw Solid Shape (Ghost)
                 glColor4f(1.0, 1.0, 1.0, 0.3)

                 
                 if type == "CUBE":
                     self.draw_solid_cube_bounds(p1, p2, (1,1,1))
                 elif type == "PYRAMID":
                     # Use the same inline logic as main render or helper
                     # For preview, simple bounds box is okay but actual shape is better
                     # Let's copy-paste the Pyramid render logic briefly or reuse if possible.
                     # Since we don't have a helper for bounds-pyramid yet (it was inline), 
                     # let's just use Cube for now to avoid code duplication, OR implement the helper.
                     # Actually, for preview, let's just show the Cube Bounds for all to indicate AREA
                     # BUT user might want to see the shape.
                     # Let's stick to Cube Bounds for now as per my previous thought, BUT add a text or color?
                     # Wait, if user says "Can't place shapes", maybe they meant it fails?
                     # Let's just fix the Cube Bounds to be generic. 
                     # Retaining Cube for all is safer for "bounds" visualization.
                     self.draw_solid_cube_bounds(p1, p2, (1,1,1))
                 elif type == "SPHERE":
                     # Sphere is defined by bounds center/radius
                     center = (np.array(p1) + np.array(p2)) / 2
                     dims = np.abs(np.array(p2) - np.array(p1)) / 2
                     glPushMatrix()
                     glTranslatef(*center)
                     glScalef(*dims)
                     gluSphere(self.quadric, 1.0, 16, 16)
                     glPopMatrix()
             
             # Reset
             self.current_preview_shape = None
             glDisable(GL_BLEND)

        # Draw all finished strokes(GL_BLEND)

        # Draw Cursor INSIDE the rotated world (so it stays synced)
        if self.cursor_pos:
            glPushMatrix()
            glTranslatef(*self.cursor_pos)
            
            # Simple Sphere for Cursor
            glColor3f(1.0, 0.0, 1.0) # Purple cursor
            glDisable(GL_LIGHTING)
            gluSphere(self.quadric, 0.15, 10, 10)
            glEnable(GL_LIGHTING)
            
            glPopMatrix()

        # Draw all finished strokes
        # Switch to material coloring
        glColor3f(1.0, 1.0, 1.0) # White base color
        
        for i, stroke in enumerate(self.lines):
            # Highlight Selection
            if i in self.selected_indices:
                glColor3f(1.0, 0.0, 0.0) # Red for selection
            else:
                glColor3f(1.0, 1.0, 1.0)
                
            if len(stroke) > 1:
                for j in range(len(stroke) - 1):
                    self.draw_cylinder(stroke[j], stroke[j+1], radius=0.03)
                    
        # Draw current stroke
        # Semi-transparent green
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 1.0, 0.0, 0.5) 
        
        if len(self.current_stroke) > 1:
            for i in range(len(self.current_stroke) - 1):
                 self.draw_cylinder(self.current_stroke[i], self.current_stroke[i+1], radius=0.03)

        # Last point of current stroke sphere cap
        if len(self.current_stroke) > 0:
             p = self.current_stroke[-1]
             glPushMatrix()
             glTranslatef(p[0], p[1], p[2])
             gluSphere(self.quadric, 0.03, 16, 16) # Increased slices 6 -> 16
             glPopMatrix()
        
        glDisable(GL_BLEND)
            
        glPopMatrix()
        # --- End World Rendering ---
        
        # Removed Cursor Rendering from here (moved inside)

        pygame.display.flip()

        
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Clean up quadric
                    gluDeleteQuadric(self.quadric)
                    pygame.quit()
                    return False
        return True
