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
        self.camera_z = -5.0

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

    def get_world_point(self, x, y, z):
        """
        Converts screen (x, y) and depth (z) to 3D World Coordinates
        using the heuristic consistent with add_point.
        """
        # Map screen coordinates to world coordinates roughly
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1) # Flip Y
        
        # Initial point in Screen Space (before rotation)
        p_screen = np.array([norm_x * 2, norm_y * 2, z * 2])
        
        # Inverse Rotation: Rotate by -rot_x (Pitch) then -rot_y (Yaw)
        rad_x = np.radians(-self.rot_x)
        rad_y = np.radians(-self.rot_y)
        
        # Rotation Matrix for X (Pitch)
        rx_mat = np.array([
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x), np.cos(rad_x)]
        ])
        
        # Rotation Matrix for Y (Yaw)
        ry_mat = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])
        
        # Apply Inverse: Inv(Ry) * Inv(Rx) * P_screen
        p_temp = np.dot(rx_mat, p_screen)
        p_final = np.dot(ry_mat, p_temp)
        
        return tuple(p_final)

    def get_interface_position(self, x, y, z):
        """
        Returns the screen pixel coordinates (sx, sy) where the 3D cursor 
        corresponding to input (x, y, z) would appear.
        """
        world_pt = self.get_world_point(x, y, z)
        return self.project_point(world_pt)

    def add_point(self, x, y, z, start_new_stroke=False):
        world_point = self.get_world_point(x, y, z)

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
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1) # Flip Y
        self.cursor_pos = (norm_x * 2, norm_y * 2, z * 2)


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

    def move_selection(self, dx, dy):
        """
        Move selected lines in screen plane (approximate).
        We map dx, dy (interface coords roughly?) or world?
        Let's assume dx, dy are World Space deltas for simplicity first, 
        or we need to map View Space delta to World Space delta.
        """
        # Simplest: Just add to X and Y (since we are drawing in 3D but logic is View-based).
        # Actually our gestures give us 'dx' 'dy' in pixel-like or arbitrary units.
        # We need to scale them to world units.
        
        scale = 0.01
        move_vec = np.array([dx * scale, -dy * scale, 0]) # Flip Y for world
        
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
        
        rad_x = np.radians(self.rot_x)
        rad_y = np.radians(self.rot_y)
        
        # Rotate Y
        rx, ry, rz = x, y, z
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
        self.selected_indices = [] # Clear selection after erase to avoid index errors

    def add_shape(self, shape_type, x, y, z, scale=0.5):
        """
        Add a 3D shape at the given position.
        Shapes are just strokes (lines) formed into the shape.
        """
        center = self.get_world_point(x, y, z)
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
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, self.camera_z) # Move camera back using dynamic Z
        
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
        
        # Apply Rotation
        glRotatef(self.rot_x, 1, 0, 0) # Rotate X
        glRotatef(self.rot_y, 0, 1, 0) # Rotate Y
        
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

        # --- Cursor Rendering (Screen/Camera Space) ---
        if self.cursor_pos:
            glPushMatrix()
            glTranslatef(*self.cursor_pos)
            
            # Simple Sphere for Cursor
            glColor3f(1.0, 0.0, 1.0) # Purple cursor
            # Disable lighting for the cursor itself so it always glows? 
            # Or keep it lit. Let's disable for it to appear "bright"
            glDisable(GL_LIGHTING)
            gluSphere(self.quadric, 0.15, 10, 10)
            glEnable(GL_LIGHTING)
            
            glPopMatrix()

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
