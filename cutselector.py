#!/usr/bin/env python3

import sys
from copy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class CutSelector:
    """ Mouse and keyboard controls for changing cuts

    A CutSelector object is bound to a Figure containing a Axes object.
    When plotting 1D cuts of a multi-variate function whose values are stored in F,
    a CutSelector allows to change the displayed cut with mouse and keyboard.
    Use the scroll wheel of the mouse to change the cut index along a given coordinate.
    Hold the shift key to scroll faster.
    Press 'up' or 'down' to change the coordinate whose index is changed.

    Future upgrades:
    - Optional display of cut variables/indices
    """
    def __init__(self, fig, axs, x, F, init_indices: list = [], order="C", show_coords=True):
        self.fig = fig
        self.axs = axs
        self.x = x
        self.F = F
        self.shape = F[0].shape
        self.Nd = len(self.shape)
        self.saved_indices = list(init_indices)
        self.order = order
        self.moving_dim = 0
        self.lines = dict()
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_press = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.cid_release = self.fig.canvas.mpl_connect('key_press_event', self.on_key_pressed)
        self.fast_scroll = False
        self.vlines = None
        self.coord_labels = None

        # Do nothing if 1d data
        if self.Nd == 1:
            return

        # Check that F data shape is consistent
        assert all([Fi.shape == self.shape for Fi in F]), "Not all elements " \
                + f"of F have the same shape {[Fi.shape for Fi in F]}."
        assert tuple([len(xi) for xi in x]) == self.shape, \
                "Inconsistent coordinates vs function values sizes, " \
                + f"expecting {self.shape}."

        # Define initial cut indices
        if not init_indices :
            self.saved_indices = [0 for _ in range(self.Nd)]
        else :
            assert len(init_indices) == self.Nd,  "CutSelector.__init__(): init_indices must be of length len(F.shape)."

        # Remember if F array is stored in row-major or column-major indexing
        assert order == "C" or order == "F", "CutSelector.__init__(): order should be either \"C\" or \"F\"."
        if order == "F":
            raise(NotImplementedError, "Sorry, only C-style row major indexing for the moment.")
        
        # Store line objects that are shown in the axes
        for ax in axs:
            self.lines[ax] = ax.get_lines()

        # Show vertical lines corresponding to the cut coordinates
        if show_coords:
            self.vlines = []
            self.coord_labels = []
            for dim, ax in enumerate(self.axs):
                the_x = self.x[dim][self.saved_indices[dim]]
                line = ax.axvline(the_x, color='k')
                self.vlines.append(line)
                label = ax.annotate("{:.3f}".format(the_x),
                        (the_x, 1),
                        ha='left', va='top',
                        xycoords=('data','axes fraction'),
                        in_layout=False
                        )
                self.coord_labels.append(label)
            self.updateCoordCut()

    def updateCoordCut(self):
        if self.vlines is None:
            return

        for dim, (line, label) in enumerate(zip(self.vlines, self.coord_labels)):
            x = self.x[dim][self.saved_indices[dim]]
            _, y = line.get_data()
            line.set_data(x, y)
            label.set_text("{:.3f}".format(x))
            label.set_x(x)
            if dim == self.moving_dim:
                line.set_linestyle('-')
            else:
                line.set_linestyle('--')

    def updateCurves(self):
        # Update graphs in each Axes object
        for dim, ax in enumerate(self.axs):
            if dim == self.moving_dim:  # ... except the one of the scrolled coordinate
                continue

            # Get flat indices of points if F appearing in the curve
            # Starting index
            start_list = deepcopy(self.saved_indices)
            start_list[dim] = 0
            start = CutSelector.flat_index(start_list, self.shape, self.order)
            
            # Ending index
            stop_list = deepcopy(self.saved_indices)
            stop_list[dim] = self.shape[dim]
            stop = CutSelector.flat_index(stop_list, self.shape, self.order)

            # Steps over flat indices
            step = 1
            if self.order == "C":
                for d in range(dim+1,len(self.shape)):
                    step *= self.shape[d]
            elif self.order == "F":
                for d in range(dim+1,len(self.shape)):
                    step *= self.shape[d]
            
            # Update lines in the Axes object
            for n,line in enumerate(self.lines[ax]):
                x, _ = line.get_data()
                y = self.F[n].flatten()[start:stop:step]
                line.set_data(x, y)

    def redraw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_key_pressed(self, event):
        if event.key == 'shift':
            self.fast_scroll = True

    def on_key_release(self, event):
        if event.key not in ['up', 'down', 'shift']:
            return

        if event.key == 'shift':
            self.fast_scroll = False
            return

        shift = 0
        if event.key == 'up':
            shift = 1
        if event.key == 'down':
            shift = -1

        temp_dim = self.moving_dim + shift
        if temp_dim >= 0 and temp_dim < len(self.shape):
            self.moving_dim = temp_dim
            self.updateCoordCut()
            self.redraw()
        
    def on_scroll(self,event):
        s = int(event.step)
        if self.fast_scroll:
            s *= (self.shape[self.moving_dim]-1)//10

        i = self.saved_indices[self.moving_dim] 
        if s<0:
            i = np.max((0, i+s))
        else:
            i = np.min((self.shape[self.moving_dim]-1, i+s))
        self.saved_indices[self.moving_dim] = i
        # print(f"Scrolled by {s}. Now i = {i}")
        self.updateCoordCut()
        self.updateCurves()
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
    x = np.linspace(x1,x2,Nx)
    y = np.linspace(y1,y2,Ny)
    z = np.linspace(z1,z2,Nz)
    X, Y, Z = np.mgrid[x1:x2:Nx*1j, y1:y2:Ny*1j, z1:z2:Nz*1j]
    F1 = X*np.cos(2*np.pi*Y)*np.exp(-0.5 * ((X/Z)**2 + (Y/Z)**2) )
    F2 = np.exp(-0.5 * ((X-Z)**2 + (Y-2*Z)**2))

    fig, axs = plt.subplots(1,3)
    
    ax = axs[0]
    ax.plot(X[:, 0, 0].flatten(), F1[:, Ny//2, Nz//2].flatten(), marker='x')
    ax.plot(X[:, 0, 0].flatten(), F2[:, Ny//2, Nz//2].flatten(), marker='+')
    ax.set_ylim((np.min(F1), np.max(F1)))
    ax.set_xlabel(r"$x$")
    
    ax = axs[1]
    ax.plot(Y[0, :, 0].flatten(), F1[Nx//2, :, Nz//2].flatten(), marker='x')
    ax.plot(Y[0, :, 0].flatten(), F2[Nx//2, :, Nz//2].flatten(), marker='+')
    ax.set_ylim((np.min(F1), np.max(F1)))
    ax.set_xlabel(r"$y$")
    
    ax = axs[2]
    ax.plot(Z[0, 0, :].flatten(), F1[Nz//2, Nz//2, :].flatten(), marker='x')
    ax.plot(Z[0, 0, :].flatten(), F2[Nz//2, Nz//2, :].flatten(), marker='+')
    ax.set_ylim((np.min(F1), np.max(F1)))
    ax.set_xlabel(r"$z$")

    for ax in axs:
        ax.set_ylabel(r"$F(x,y,z)$")
    cs = CutSelector(fig, axs, (x,y,z), (F1,F2), [Nx//2, Ny//2, Nz//2])

    plt.show()

if __name__ == '__main__':
    main(sys.argv)
