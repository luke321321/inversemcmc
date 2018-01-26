# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:33:32 2018

@author: s1002685
"""

import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays, array_shapes

from GaussianProcess import GaussianProcess
  


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
        
def test_r2_distance_tests(x, y):
    GP = GaussianProcess(np.array([0,1]), np.array([0,1]))
    d_xy = GP.r2_distance(x ,y)
    d_yx = GP.r2_distance(y, x)
    
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
    test_r2_distance_list_vectors()
    test_r2_distance_vectors()
    test_r2_distance_mixed()