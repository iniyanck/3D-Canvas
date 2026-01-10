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
        self.lines = []
        self.current_stroke = []
        # Optional: Reset rotation on clear?
        # self.rot_x = 0
        # self.rot_y = 0

    def rotate(self, d_yaw, d_pitch):
        """
        Accumulate rotation angles.
        d_yaw: Change in Y rotation (horizontal movement)
        d_pitch: Change in X rotation (vertical movement)
        """
        self.rot_y += d_yaw
        self.rot_x += d_pitch

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
        eraser_pos = np.array([screen_x, screen_y])
        new_lines = []
        
        # Also check current stroke
        if self.current_stroke:
             self.lines.append(self.current_stroke)
             self.current_stroke = []

        for stroke in self.lines:
            new_stroke = []
            stroke_points_screen = []
            
            # Pre-calculate screen points for this stroke to avoid re-projecting repeatedly?
            # Or just do it on the fly. Doing it on the fly is simpler to implement logic.
            
            # We iterate point by point.
            # Ideally we check segments.
            
            if len(stroke) < 2:
                # Single point stroke
                p_screen = self.project_point(stroke[0])
                if p_screen:
                    dist = np.linalg.norm(np.array(p_screen) - eraser_pos)
                    if dist > radius:
                         new_lines.append(stroke)
                continue

            # Segment based erasure
            # We rebuild the stroke.
            current_segment = []
            
            for i in range(len(stroke)):
                p3d = stroke[i]
                p2d = self.project_point(p3d)
                
                # If point is behind camera, we can't really erase it properly visually, keep it?
                # Or assume it's not interactable.
                if p2d is None:
                    current_segment.append(p3d)
                    continue
                
                p2d_arr = np.array(p2d)
                dist_point = np.linalg.norm(p2d_arr - eraser_pos)
                
                # Check point itself
                if dist_point <= radius:
                    # Point is erased. Break segment.
                    if current_segment:
                        if len(current_segment) > 1: # Only save valid lines? or points too?
                             # Actually keep single points if they are leftovers
                             new_lines.append(current_segment)
                        elif len(current_segment) == 1:
                             # Keep single dots
                             new_lines.append(current_segment)
                        current_segment = []
                    continue # Skip adding this point
                
                # Check segment leading to this point
                segment_erased = False
                if current_segment:
                    prev_p3d = current_segment[-1]
                    prev_p2d = self.project_point(prev_p3d)
                    
                    if prev_p2d is not None:
                        dist_seg = self.get_distance_to_segment_2d(prev_p2d, p2d, eraser_pos)
                        if dist_seg <= radius:
                            segment_erased = True
                
                if segment_erased:
                     # Split here.
                     # The previous segment is valid up to the cut. 
                     # Simplifying: If a segment is hit, we break the line. 
                     # We already added the previous point to current_segment.
                     # So we save current_segment and start fresh.
                     # Ideally we could calculate exact cut intersection, but splitting at vertex is easier.
                     if len(current_segment) > 0:
                         new_lines.append(current_segment)
                     current_segment = [p3d] # Start new, but wait... 
                     # If the segment is erased, the geometric connection is broken.
                     # We shouldn't add the *connection* but we checked the point was safe.
                     # So p3d is a valid start for a new segment.
                     pass 
                else:
                    current_segment.append(p3d)
            
            if len(current_segment) > 0:
                new_lines.append(current_segment)
                
        self.lines = new_lines

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
        
        for stroke in self.lines:
            if len(stroke) > 1:
                for i in range(len(stroke) - 1):
                    self.draw_cylinder(stroke[i], stroke[i+1], radius=0.03)
                    
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
