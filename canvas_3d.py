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

        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Gesture Canvas")

        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5) # Move camera back

    def add_point(self, x, y, z, start_new_stroke=False):
        # Map screen coordinates to world coordinates roughly
        # This is a simplification; for true raycasting we'd need more math,
        # but for a simple "air canvas" this works.
        
        # Normalize x, y to -1 to 1 range
        norm_x = (x / self.width) * 2 - 1
        norm_y = -((y / self.height) * 2 - 1) # Flip Y
        
        # Scale z appropriately. MediaPipe z is roughly -0.1 to 0.1 depending on hand size relative to cam?
        # Actually MediaPipe z is relative to wrist. 
        # Let's trust the z passed in is reasonable or scale it.
        # We might need to calibrate this. For now let's just take it as is.
        
        world_point = (norm_x * 2, norm_y * 2, z * 2) # *2 to widen field of view usage

        if start_new_stroke:
            if self.current_stroke:
                self.lines.append(self.current_stroke)
                self.current_stroke = []
        
        self.current_stroke.append(world_point)
        # Also store all points for point cloud view? maybe not needed if we draw lines.

    def clear(self):
        self.lines = []
        self.current_stroke = []

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
        
        # Draw all finished strokes
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0) # White lines
        
        for stroke in self.lines:
            if len(stroke) > 1:
                for i in range(len(stroke) - 1):
                    glVertex3fv(stroke[i])
                    glVertex3fv(stroke[i+1])
        
        # Draw current stroke
        if len(self.current_stroke) > 1:
            glColor3f(0.0, 1.0, 0.0) # Green for active stroke
            for i in range(len(self.current_stroke) - 1):
                glVertex3fv(self.current_stroke[i])
                glVertex3fv(self.current_stroke[i+1])
        
        glEnd()

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
