#!/usr/bin/env python3

import sys
from copy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class CutSelector:
    def __init__(self, ax, fx, axis=0, init_indices=None, order="C"):
        self.ax = ax
        self.fx = fx
        self.shape = fx.shape
        assert axis >= 0 and axis < len(fx.shape),  "CutSelector.__init__(): axis should be 0 <= axis < {}".format(len(fx.shape))
        self.axis_dim = axis

        if init_indices is None :
            self.saved_indices = [0 for _ in range(len(fx.shape))]
        else :
            assert len(init_indices) == len(fx.shape),  "CutSelector.__init__(): init_indices must be of length len(fx.shape)."
            self.saved_indices = init_indices
            self.saved_indices[axis] = -1

        assert order == "C" or order == "F", "CutSelector.__init__(): order should be either \"C\" or \"F\"."
        self.order = order

        for d,si in enumerate(self.saved_indices):
            if si != -1:
                self.moving_dim = d
                break

        self.line, = ax.get_lines()
        self.cid_scroll = self.line.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_press = self.line.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

    def redraw(self):
        indices = self.saved_indices

        start_list = deepcopy(indices)
        start_list[start_list.index(-1)] = 0
        start = CutSelector.flat_index(start_list, self.shape, self.order)
        
        stop_list = deepcopy(indices)
        stop_list[stop_list.index(-1)] = self.shape[self.axis_dim]
        stop = CutSelector.flat_index(stop_list, self.shape, self.order)

        step = 1
        if self.order == "C":
            for dim in range(self.axis_dim+1,len(self.shape)):
                step *= self.shape[dim]
        elif self.order == "F":
            for dim in range(self.axis_dim+1,len(self.shape)):
                step *= self.shape[dim]

        x, _ = self.line.get_data()
        y = self.fx.flatten()[start:stop:step]
        self.line.set_data(x, y)
        self.line.figure.canvas.draw()

    def on_key_release(self, event):
        if event.key not in ['up', 'down']:
            return

        shift = 0
        if event.key == 'up':
            shift = 1
        if event.key == 'down':
            shift = -1

        temp_dim = self.moving_dim + shift
        if temp_dim == self.axis_dim :
            temp_dim += shift

        if temp_dim >= 0 and temp_dim < len(self.shape):
            self.moving_dim = temp_dim

        # print(f"Now controlling {self.moving_dim}")
        
    def on_scroll(self,event):
        s = int(event.step)
        i = self.saved_indices[self.moving_dim] 
        if s<0:
            i = np.max((0, i+s))
        else:
            i = np.min((self.shape[self.moving_dim]-1, i+s))
        self.saved_indices[self.moving_dim] = i
        # print(f"Scrolled by {s}. Now i = {i}")
        self.redraw()

    @staticmethod
    def flat_index(indices, shape, order='C'):
        assert(len(indices) == len(shape))
        dims = len(shape)
        I = 0
        for d in range(dims):
            prod_N = 1
            if order == "C":
                for k in range(d+1,dims) :
                    prod_N *= shape[k]
            if order == "F":
                for k in range(d) :
                    prod_N *= shape[k]
            I += prod_N * indices[d]
        return I
            
def main(argv):
    x1, x2, y1, y2, z1, z2 = (-3,3,-2,2,0.1,3)
    Nx, Ny, Nz = (51,51,21)
    X, Y, Z = np.mgrid[x1:x2:Nx*1j, y1:y2:Ny*1j, z1:z2:Nz*1j]
    F = X*np.cos(2*np.pi*Y)*np.exp(-0.5 * ((X/Z)**2 + (Y/Z)**2) )

    fig, ax = plt.subplots()
    ax.plot(X[:, 0, 0].flatten(), F[:, Ny//2, Nz//2].flatten(), marker='x')
    ax.set_ylim((np.min(F), np.max(F)))
    ax.set_xlabel(r"$x$")
    cs = CutSelector(ax, F, 0, [0, Ny//2, Nz//2])

    fig2, ax2 = plt.subplots()
    ax2.plot(Y[0, :, 0].flatten(), F[Nx//2, :, Nz//2].flatten(), marker='x')
    ax2.set_ylim((np.min(F), np.max(F)))
    ax2.set_xlabel(r"$y$")
    cs2 = CutSelector(ax2, F, 1, [Nx//2, 0, Nz//2])

    plt.show()

if __name__ == '__main__':
    main(sys.argv)
