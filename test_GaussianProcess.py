# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:33:32 2018

@author: s1002685
"""

import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats

from GaussianProcess import GaussianProcess
  
dist_array = arrays(np.float64, array_shapes(2,2,1,100), elements=floats(min_value=-1e30, max_value=1e30)) 
@given(dist_array, dist_array, dist_array)
def test_r2_distance(x, y, z):
    GP = GaussianProcess(np.array([0,1]), np.array([0,1]))
    
    #if dim x,y > 1 then need x.shape[1] == y.shape[1]
    if len(x.shape) > 1 and len(y.shape) > 1:
        assume(x.shape[1] == y.shape[1])
    if len(x.shape) > 1 and len(z.shape) > 1:
        assume(x.shape[1] == z.shape[1])
    if len(y.shape) > 1 and len(z.shape) > 1:
        assume(y.shape[1] == z.shape[1])
    
    d_xy = GP.r2_distance(x,y)
    d_yx = GP.r2_distance(y,x)
    d_xz = GP.r2_distance(x,z)
    d_zy = GP.r2_distance(z,y)
    
    #check triangle inequality
    assert np.amax(np.sqrt(d_xy)) <= np.amax(np.sqrt(d_xz)) + np.amax(np.sqrt(d_zy))
    
    #Check symmetry
    assert np.array_equiv(d_xy, d_yx.T)
    
    #check d(x,0) = 0 iff x = 0
    assert (GP.r2_distance(x, np.zeros(x.shape)) == 0).all() == (np.sum(np.abs(x)) == 0)
    
    #d(x,x) == 0
    d_xx = GP.r2_distance(x, x)
    if len(d_xx.shape) <= 1:
        assert (d_xx == 0).all()
    else:
        assert (d_xx.diagonal() == 0).all()
    
if __name__ == '__main__':
    test_r2_distance()