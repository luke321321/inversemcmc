# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:33:32 2018

@author: s1002685
"""

import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays, array_shapes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GaussianProcess import GaussianProcess
  
def test_r2_distance():
    test_r2_distance_list_vectors()
    test_r2_distance_vectors()
    test_r2_distance_mixed()
    test_r2_distance_floats()

vec = arrays(np.float64, array_shapes(1,1,1,100), elements=st.floats(min_value=-1e15, max_value=1e15)) 
mixed = arrays(np.float64, array_shapes(1,2,2,100), elements=st.floats(min_value=-1e15, max_value=1e15)) 

@given(st.data())
def test_r2_distance_list_vectors(data):
    #draw both x and y so that have shapes (m x d) and (n x d) respectively
    d = data.draw(st.integers(1,100), 'd')
    strat_array = arrays(np.float64, (10, d), elements=st.floats(min_value=-1e5, max_value=1e5)) 
    x = data.draw(strat_array, 'x')
    y = data.draw(strat_array, 'y')

    test_r2_distance_tests(x, y)

@given(vec , vec)
def test_r2_distance_vectors(x, y):
    test_r2_distance_tests(x, y)
    
@given(vec , mixed)
def test_r2_distance_mixed(x, y):
    test_r2_distance_tests(x, y)
    
@given(vec , mixed)
def test_r2_distance_floats(x, y):
    test_r2_distance_tests(x[0], y)
        
def test_r2_distance_tests(x, y):
    d_xy = GaussianProcess.r2_distance(x ,y)
    d_yx = GaussianProcess.r2_distance(y, x)
    
    #Check symmetry
    assert np.array_equiv(d_xy, d_yx.T) or np.array_equiv(d_xy, d_yx)
    
    #check d(x,0) = 0 iff x = 0
    assert (GaussianProcess.r2_distance(x, np.zeros(x.shape)) == 0).all() == (np.sum(np.abs(x)) == 0)
    
    #d(x,x) == 0
    d_xx = GaussianProcess.r2_distance(x, x)
    if len(d_xx.shape) <= 1:
        assert (d_xx == 0).all()
    else:
        assert (d_xx.diagonal() == 0).all()
        
def test_brownian_bridge():
    points = np.array([0,0.5,1])
    data = np.array([0,0,1.])
    plt.figure()
    for i in range(30):
        data[1] = GaussianProcess.Brownian_bridge(points,data,0.5)
        plt.plot(points, data)
    plt.show()
        

def test_GP_interp_1d():
    design_pts = GaussianProcess.create_uniform_grid(-2,2,10)
    obs = np.random.normal(size = 10)
    GP1 = GaussianProcess(design_pts, obs)
    GP_interp_method = GP1.GP(100)
    vGP_interp_method = np.vectorize(GP_interp_method)
    
    #Plot results:
    plt.figure()
    plt.plot(design_pts, obs, 'ro')
    grid = GaussianProcess.create_uniform_grid(-2,2,1000)
    plt.plot(grid, vGP_interp_method(grid))
    plt.show()

def test_GP_interp_2d():
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5,2)
    obs = np.random.randn(5 ** 2)
    GP1 = GaussianProcess(design_pts, obs)
    GP_interp_method = GP1.GP(25)
    vGP_interp_method = np.vectorize(GP_interp_method, signature='(i)->()')
    
    #Plot results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = GaussianProcess.create_uniform_grid(-2,2,50,2)
    X = Z[:,0]
    Y = Z[:,1]
    #Plot the surface
    ax.plot_trisurf(X, Y, vGP_interp_method(Z)) #, antialiased=True
    #Plot the design points
    ax.scatter(design_pts[:,0], design_pts[:,1], obs, color='green')
    plt.show()
    
if __name__ == '__main__':
    test_r2_distance()
#    test_brownian_bridge()
#    test_GP_interp_1d()
#    test_GP_interp_2d()