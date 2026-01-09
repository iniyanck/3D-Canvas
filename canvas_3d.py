import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

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


        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Gesture Canvas")

        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, self.camera_z) # Move camera back

    def add_point(self, x, y, z, start_new_stroke=False):
        # Map screen coordinates to world coordinates roughly
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1) # Flip Y
        
        # Initial point in Screen Space (before rotation)
        # We want to find P_world such that Rotation * P_world = P_screen
        # So P_world = InverseRotation * P_screen
        
        p_screen = np.array([norm_x * 2, norm_y * 2, z * 2])
        
        # Inverse Rotation: Rotate by -rot_x (Pitch) then -rot_y (Yaw)
        # Convert to radians
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
        
        world_point = tuple(p_final)

        if start_new_stroke:
            if self.current_stroke:
                self.lines.append(self.current_stroke)
                self.current_stroke = []
        
        self.current_stroke.append(world_point)
        # Also store all points for point cloud view? maybe not needed if we draw lines.

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

    def erase_at(self, x, y, z, radius=0.1):
        # Normalize x, y to -1 to 1 range
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1) # Flip Y
        eraser_point = np.array([norm_x * 2, norm_y * 2, z * 2])
        
        new_lines = []
        
        # Check current stroke first
        if self.current_stroke:
             # If current stroke is being erased, just finish it and push it to lines to be processed
             self.lines.append(self.current_stroke)
             self.current_stroke = []

        for stroke in self.lines:
            new_stroke = []
            for point in stroke:
                p = np.array(point)
                dist = np.linalg.norm(p - eraser_point)
                
                # We ignore Z depth for easier erasing for now, or use a larger Z tolerance
                # Let's use 2D distance for X/Y and a loose Z check
                dist_xy = np.linalg.norm(p[:2] - eraser_point[:2])
                dist_z = abs(p[2] - eraser_point[2])
                
                # If point is OUTSIDE radius, keep it
                if dist_xy > radius: # or dist_z > 0.5 (optional depth check)
                    new_stroke.append(point)
                else:
                    # Point is erased.
                    # If we have a gathered stroke segment, save it and start a new one
                    if new_stroke:
                        if len(new_stroke) > 1:
                            new_lines.append(new_stroke)
                        new_stroke = []
            
            # Append remaining part of the stroke
            if new_stroke and len(new_stroke) > 1:
                new_lines.append(new_stroke)
                
        self.lines = new_lines

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity() # Reset view matrix
        
        # Camera Setup
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, self.camera_z) # Move camera back using dynamic Z
        
        # --- World Rendering (Rotated) ---
        glPushMatrix()
        
        # Apply Rotation
        glRotatef(self.rot_x, 1, 0, 0) # Rotate X
        glRotatef(self.rot_y, 0, 1, 0) # Rotate Y
        
        # Draw all finished strokes
        glLineWidth(1)
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0) # White lines
        
        for stroke in self.lines:
            if len(stroke) > 1:
                for i in range(len(stroke) - 1):
                    glVertex3fv(stroke[i])
                    glVertex3fv(stroke[i+1])
        glEnd()
        
        # Draw current stroke
        glLineWidth(2)
        if len(self.current_stroke) > 1:
            glBegin(GL_LINES)
            glColor3f(0.0, 1.0, 0.0) # Green for active stroke
            for i in range(len(self.current_stroke) - 1):
                glVertex3fv(self.current_stroke[i])
                glVertex3fv(self.current_stroke[i+1])
            glEnd()
            
        glPopMatrix()
        # --- End World Rendering ---

        # --- Cursor Rendering (Screen/Camera Space) ---
        if self.cursor_pos:
            glPushMatrix()
            # Cursor pos is already relative to camera (computed from screen xy + z)
            # cursor_pos is (norm_x*2, norm_y*2, z*2)
            # It's in the same "camera space" as the world before rotation.
            glTranslatef(*self.cursor_pos)
            
            # Dynamic Size based on Z
            # self.cursor_pos[2] is the Z value. 
            # Closer (positive Z relative to -5 offset) -> Bigger
            # Farther (negative Z) -> Smaller
            # Our camera is at Z=0 looking at -5? No, we translated(0,0,-5).
            # So objects are at roughly 0,0,0 relative to that frame.
            # cursor_pos z is roughly -2 to +2.
            
            z_val = self.cursor_pos[2]
            # Base size 10. Scale factor.
            # If Z is +1 (closer), size should be larger. 
            # 10 + (Z * 5)
            
            size = max(5.0, 10.0 + (z_val * 4.0))
            
            glPointSize(size)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 1.0) # Purple cursor
            glVertex3f(0, 0, 0)
            glEnd()
            
            glPopMatrix()

        pygame.display.flip()

        
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
        return True
