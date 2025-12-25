import numpy as np
import time
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from pyrr import Quaternion, matrix44
from icosph_nn.icosphere import Icosphere
from icosph_nn import utils

class Mesh:
    def __init__(self, vertices, colors, indices):
        self.vertex_vbo = vbo.VBO(np.array(vertices, dtype=np.float32), usage=GL_STATIC_DRAW)
        self.color_vbo = vbo.VBO(np.array(colors, dtype=np.float32), usage=GL_STATIC_DRAW)
        self.index_vbo = vbo.VBO(np.array(indices, dtype=np.uint32), usage=GL_STATIC_DRAW, target=GL_ELEMENT_ARRAY_BUFFER)
        self.indices_len = len(indices)
    
    def update_colors(self, colors):
        self.color_vbo.set_array(np.array(colors, dtype=np.float32))

    def render(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        self.vertex_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)

        self.color_vbo.bind()
        glColorPointer(3, GL_FLOAT, 0, None)

        self.index_vbo.bind()

        glDrawElements(GL_TRIANGLES, self.indices_len * 3, GL_UNSIGNED_INT, None)

        self.index_vbo.unbind()
        self.color_vbo.unbind()
        self.vertex_vbo.unbind()

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
    
    def unload(self):
        self.vertex_vbo.delete()
        self.color_vbo.delete()
        self.index_vbo.delete()

class IcosphereVisualizer:
    def __init__(self, level, fix_rotation_axis = False, width=800, height=600, fov=60, title="Icosphere visualizer"):
        
        self._level = level
        self._mesh = None
        self._width = width
        self._height = height
        self._last_frame_time = time.time()
        self._fov = fov
        self._dragging_start = None
        self._wireframe = False
        self.fix_rotation_axis = fix_rotation_axis

        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(self._width, self._height)
        glutCreateWindow(title.encode('utf-8'))

        # Render settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)

        glutDisplayFunc(self._display)
        glutIdleFunc(self._display)
        glutReshapeFunc(self._resize)

        glutMouseFunc(self._mouse_click)
        glutMotionFunc(self._mouse_move)
        glutKeyboardUpFunc(self._key_release)
        glutSpecialUpFunc(self._key_release)

        self.mesh_orientation = Quaternion()
        self.rotation_axis = np.array((0, 1, 0), dtype=np.float32)
        self.rotation_speed = 10

        self.on_key_release = None

    def change_level(self, new_level):
        if self._level != new_level:
            self.remove_mesh()
            self._level = new_level
            icosphere = Icosphere(new_level)
            self.vertices = icosphere.generate_vertices()
            self.triangles = icosphere.generate_triangles

    def remove_mesh(self):
        if self._mesh is not None:
            self._mesh.unload()
            self._mesh = None

    def update_mesh(self, colors):
        colors = np.array(colors, dtype=np.float32)
        assert len(colors.shape) == 2
        assert utils.get_icosphere_level(colors.shape[0]) == self._level
        assert colors.shape[1] == 3

        if self._mesh is not None:
            self._mesh.update_colors(colors)
        else:
            ico = Icosphere(self._level)
            self._mesh = Mesh(ico.generate_vertices(), colors, ico.generate_triangles())

    def update_title(self, title):
        glutSetWindowTitle(title.encode('utf-8'))

    def main_loop(self):
        glutMainLoop()
    
    def _display(self):
        now = time.time()
        delta_time = now - self._last_frame_time
        self._last_frame_time = now

        # apply rotation
        q = Quaternion.from_axis_rotation(self.rotation_axis, math.radians(self.rotation_speed * delta_time))
        self.mesh_orientation = (self.mesh_orientation * q).normalised
        
        # slow down rotation speed
        self.rotation_speed = max(self.rotation_speed - 1 * self.rotation_speed * delta_time, 0)
 
        glClearColor(0.9, 0.9, 0.9, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()    
        gluLookAt(0, 0, -3, 0, 0, 0, 0, 1, 0)
        glPushMatrix()
        glMultMatrixf(matrix44.create_from_quaternion(self.mesh_orientation).astype(np.float32))

        if self._mesh is not None:
            self._mesh.render()

        glPopMatrix()

        glutSwapBuffers()

    def _key_release(self, key, x, y):  
        if key == b'w':
            self._wireframe = not self._wireframe
            if self._wireframe: 
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.on_key_release is not None:
            self.on_key_release(key)    

    def _mouse_click(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self.rotation_speed = 0
            self._dragging_start = (x, y)
        if button == GLUT_LEFT_BUTTON and state == GLUT_UP:
            self._dragging_start = None
        if button == 3:
            self._fov *= 0.99
            self._resize(self._width, self._height)
        if button == 4:
            self._fov *= 1.01
            self._resize(self._width, self._height)

    def _mouse_move(self, x, y):
        if self._dragging_start:
            sx, sy = self._dragging_start
            dx = x - sx
            dy = y - sy
            if dx == 0 and dy == 0:
                return

            if self.fix_rotation_axis:
                self.rotation_axis = np.array((0, math.copysign(1, -dx), 0), np.float32)
                self.rotation_speed = math.fabs(dx) * self._fov / 60
            else:
                axis = np.array((dy, -dx, 0), dtype=np.float32)
                length = np.linalg.norm(axis)
                self.rotation_axis = axis / length
                self.rotation_speed = length * self._fov / 60

    def _resize(self, width, height):
        if width == 0 or height == 0:
            return

        self._width = width
        self._height = height

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self._fov, width/height, 0.1, 100)