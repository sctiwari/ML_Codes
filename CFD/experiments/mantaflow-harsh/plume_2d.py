#
# Simple example scene for a 2D simulation
# Simulation of a buoyant smoke density plume with open boundaries at top & bottom

from datetime import datetime
import os

import numpy as np
from manta import *

class Plume2DScene(object):

    def __init__(self, res=64, enable_gui=True):
        # solver params
        self.res = res
        self.gs = vec3(res, res, 1)
        self.s = Solver(name='main', gridSize=self.gs, dim=2)
        self.s.timestep = 1.0
        self.timings = Timings()

        self.sess = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_res_{self.res}"
        self.sess_dir = f'data/{self.sess}'
        os.makedirs(self.sess_dir)

        # prepare grids
        self.flags = self.s.create(FlagGrid)
        self.vel = self.s.create(MACGrid)
        self.density = self.s.create(RealGrid)
        self.pressure = self.s.create(RealGrid)

        self.b_width = 1
        self.flags.initDomain(boundaryWidth=self.b_width)
        self.flags.fillGrid()

        open_bounds = "".join(np.random.choice(['x', 'X', 'y', 'Y'], np.random.randint(0, 4), replace=False))
        print(open_bounds)
        setOpenBound(self.flags, self.b_width, open_bounds, FlagOutflow | FlagEmpty)

        self.source = self.s.create(
            Sphere, center=self.gs*vec3(*np.random.normal(0.5, scale=0.2, size=3)), radius=res* np.random.uniform(0.01, 0.1))
        self.source_enable_time = np.random.uniform(0.7, 1.0)

        self._add_objects()

        if enable_gui:
            gui = Gui()
            gui.show(True)

    def _step(self, t, num_steps):
        if t == 0:
            self.sin_bouyancy_phase = np.random.uniform(-np.pi, np.pi, size=2)
            self.sin_bouyancy_freq = np.random.uniform(0, 0.05, size=2)

        self.source.applyToGrid(grid=self.vel, value=vec3(*np.sin(self.sin_bouyancy_phase + self.sin_bouyancy_freq * t), 0))
        if t / num_steps < self.source_enable_time:
            self.source.applyToGrid(grid=self.density, value=1)

        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.density, order=2) 
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.vel, order=2, openBounds=True, boundaryWidth=self.b_width)
        resetOutflow(flags=self.flags,real=self.density)

        setWallBcs(flags=self.flags, vel=self.vel)
        
        gravity = 4e-3 * vec3(*np.sin(self.sin_bouyancy_phase + self.sin_bouyancy_freq * t), 0)
        addBuoyancy(density=self.density, vel=self.vel, gravity=gravity, flags=self.flags)

        solvePressure(flags=self.flags, vel=self.vel, pressure=self.pressure)
        self.s.step()

    def _add_objects(self):
        shapes = ['box', 'sphere', 'cylinder']
        for _ in range(150):
            shape_type = np.random.choice(shapes)
            if shape_type == 'box':
                position = self.gs * vec3(*np.random.uniform(size=(3,)))
                size = self.gs * vec3(*np.random.uniform(0.0, 0.05, size=(3,)))
                obs = Box(
                    parent=self.s,
                    p0=position - size,
                    p1=position + size)
                obs.applyToGrid(grid=self.flags, value=FlagObstacle)
            elif shape_type == 'cylinder':
                obs = Cylinder(
                    parent=self.s,
                    center=self.gs * vec3(*np.random.uniform(size=(3,))),
                    radius=self.res * np.random.uniform(0, 0.05),
                    z=self.gs*vec3(*np.random.uniform(0, 0.05, size=(3,))))
                obs.applyToGrid(grid=self.flags, value=FlagObstacle)
            elif shape_type == 'sphere':
                obs = Sphere(
                    parent=self.s, center=self.gs * vec3(*np.random.uniform(size=(3,))),
                    radius=self.res * np.random.uniform(0, 0.05))
                obs.applyToGrid(grid=self.flags, value=FlagObstacle)

    def _persist(self, t):
        np_pressure = np.zeros([int(self.gs.z), int(self.gs.y), int(self.gs.x), 1])
        copyGridToArrayReal(source=self.pressure, target=np_pressure)
        np.save(f'{self.sess_dir}/pressure-{t}.npy', np_pressure)

        np_density = np.zeros([int(self.gs.z), int(self.gs.y), int(self.gs.x), 1])
        copyGridToArrayReal(source=self.density, target=np_density)
        np.save(f'{self.sess_dir}/density-{t}.npy', np_density)

        np_vel = np.zeros([int(self.gs.z), int(self.gs.y), int(self.gs.x), 3])
        copyGridToArrayVec3(source=self.vel, target=np_vel)
        np.save(f'{self.sess_dir}/velocity-{t}.npy', np_vel)

    def run(self, num_steps, persist=True):
        for t in range(num_steps):
            self._step(t, num_steps)
            if persist:
                self._persist(t)

if __name__ == '__main__':

    import sys
    print(sys.argv)

    scene = Plume2DScene(res=256)
    scene.run(350)
