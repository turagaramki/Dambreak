from __future__ import print_function
from numpy import ones_like,mgrid


from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation

from pysph.solver.application import Application

import numpy as np
import matplotlib.pyplot as plt


def Geometry():
    """ Create particles """
    plt.close('all')
    dx=0.05
    x,y=np.mgrid[-5.0:5.0:dx*.75,-5.0:5.0:dx*.75]
    xw=x.ravel()
    yw=y.ravel()
    plt.axis('equal')
    #plt.scatter(x,y)
    #plt.show()
    indices=[]
    for i in range(len(xw)):
        if (xw[i] > -4.9) & (xw[i] < 4.9):
            if (yw[i]>-4.9):
                indices.append(i)
    ro=1000
    hdx=1.2
    m=ones_like(xw)*dx*dx*ro
    h=ones_like(xw)*hdx*dx
    rho=ones_like(xw)*ro



    wall=get_particle_array_wcsph(x=xw,y=yw,m=m,rho=rho,h=h,name='wall')
    wall.remove_particles(indices)

    print ("Number of Solid particles are %d" %wall.get_number_of_particles())

    x,y=np.mgrid[-4.8:-2.9:dx,-4.8:-0.9:dx]
    xf=x.ravel()
    yf=y.ravel()

    m=ones_like(xf)*dx*dx*ro
    h=ones_like(xf)*hdx*dx
    rho=ones_like(xf)*ro

    fluid=get_particle_array_wcsph(x=xf,y=yf,m=m,rho=rho,h=h,name='fluid')
    print ("Number of fluid particles are %d" %fluid.get_number_of_particles())
    # plt.scatter(wall.x,wall.y)
    # plt.scatter(fluid.x,fluid.y)
    # plt.show()
    return [wall,fluid]


class DamBreak(Application):

    def initialize(self):
        self.co=np.sqrt(2*9.81*3)
        self.alpha=0.01

    def create_particles(self):
        """ Create particles"""
        wall,fluid=Geometry()

        #self.scheme.setup_properties([pa])

        plt.scatter(wall.x,wall.y)
        plt.scatter(fluid.x,fluid.y)
        plt.show()
        return [fluid,wall]

    def create_solver(self):
        kernel=CubicSpline(dim=2)
        integrator=EPECIntegrator(fluid=WCSPHStep(),wall=WCSPHStep())

        dt=5e-4
        tf=4.0
        solver=Solver(kernel=kernel,dim=2,integrator=integrator,dt=dt,tf=tf)

        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    TaitEOS(
                        dest='fluid', sources=None, rho0=1000,
                        c0=self.co, gamma=7.0
                    ),
                    TaitEOS(
                        dest='wall', sources=None, rho0=1000,
                        c0=self.co, gamma=7.0
                    ),
                ],
                real=False
            ),
            Group(equations=[
                ContinuityEquation(dest='fluid',  sources=['fluid', 'wall']),
                ContinuityEquation(dest='wall',  sources=['fluid', 'wall']),

                MomentumEquation(
                    dest='fluid', sources=['fluid', 'wall'],
                    alpha=self.alpha, beta=0.0, c0=self.co, gy=-9.81
                ),

                XSPHCorrection(dest='fluid', sources=['fluid']),

            ]),
        ]
        return equations

if __name__=='__main__':
    app=DamBreak()
    app.run()
