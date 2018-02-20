import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume, settings
from hypothesis.extra.numpy import arrays, array_shapes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GaussianProcess import GaussianProcess

def test_r2_distance():
    test_r2_distance_list_vectors()
    test_r2_distance_vectors()
    test_r2_distance_mixed()
    test_r2_distance_floats()

vec = arrays(np.float64, array_shapes(1,1,1,500), elements=st.floats(min_value=-1e15, max_value=1e15)) 
mixed = arrays(np.float64, array_shapes(1,2,2,500), elements=st.floats(min_value=-1e15, max_value=1e15)) 

@st.composite
def strat_shape(draw, d):
    n = draw(st.integers(1,100))
    return (n, d)

#Allow more time for these tests
with settings(deadline=None):
    @given(st.data())
    def test_r2_distance_list_vectors(data):
        #draw both x and y so that have shapes (m x d) and (n x d) respectively
        d = data.draw(st.integers(1,100), 'd')
        strat_array = arrays(np.float64, strat_shape(d), elements=st.floats(min_value=-1e5, max_value=1e5)) 
        x = data.draw(strat_array, 'x')
        y = data.draw(strat_array, 'y')
        
        test_r2_distance_tests(x, y)
    
    @given(vec , vec)
    def test_r2_distance_vectors(x, y):
        test_r2_distance_tests(x, y)
        
    @given(st.data())
    def test_r2_distance_mixed(data):
        #draw both x and y so that have shapes (d) and (n x d) respectively
        d = data.draw(st.integers(1,100), 'd')
        strat_array_x = arrays(np.float64, strat_shape(d), elements=st.floats(min_value=-1e5, max_value=1e5))
        strat_array_y = arrays(np.float64, d, elements=st.floats(min_value=-1e5, max_value=1e5))
        x = data.draw(strat_array_x, 'x')
        y = data.draw(strat_array_y, 'y')
        
        test_r2_distance_tests(x, y)
        test_r2_distance_tests(y, x)
        
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
        
def test_GP_bridge():
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5,1)
    obs = np.random.randn(5)
    GP = GaussianProcess(design_pts, obs)
    x = np.array([0.5])
    points = np.array([[0],x,[1.]])
    data = np.array([0,0,1.])
    plt.figure()
    for i in range(15):
        data[1] = GP.GP_bridge(x, points[[0,2]], data[[0,2]])
        plt.plot(points, data)
    plt.title('test_GP_bridge')
    plt.show()
    
        

def test_GP_interp_1d(plotFlag = True):
    design_pts = GaussianProcess.create_uniform_grid(-2,2,10)
    obs = np.random.normal(size = 10)
    GP1 = GaussianProcess(design_pts, obs)
    GP_interp_method = GP1.GP_interp(100)
    vGP_interp_method = np.vectorize(GP_interp_method)
    
    #Plot results:
    if plotFlag:
        plt.figure()
        plt.plot(design_pts, obs, 'ro')
        grid = GaussianProcess.create_uniform_grid(-2,2,1000)
        plt.plot(grid, vGP_interp_method(grid))
        plt.title('test_GP_interp_1d')
        plt.show()

def test_GP_interp_2d(plotFlag = True):
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5,2)
    obs = np.random.randn(5 ** 2)
    GP1 = GaussianProcess(design_pts, obs)
    GP_interp_method = GP1.GP_interp(25)
    vGP_interp_method = np.vectorize(GP_interp_method, signature='(i)->()')
    
    #Plot results:
    if plotFlag:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        Z = GaussianProcess.create_uniform_grid(-2,2,50,2)
        X = Z[:,0]
        Y = Z[:,1]
        #Plot the surface
        ax.plot_trisurf(X, Y, vGP_interp_method(Z)) #, antialiased=True
        #Plot the design points
        ax.scatter(design_pts[:,0], design_pts[:,1], obs, color='green')
        plt.title('test_GP_interp_2d')
        plt.show()
    
def test_GP_samples_1d(plotFlag = True):
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5,1)
    obs = np.random.randn(5)
    GP1 = GaussianProcess(design_pts, obs)
    grid = GaussianProcess.create_uniform_grid(-2,2,3000,1)
    realisation = GP1.GP_at_points(grid, num_evals=1).T

    if plotFlag:
        plt.figure()
        plt.plot(grid, realisation)
        plt.plot(design_pts, obs, 'ro')
        plt.title('test_GP_samples_1d')
        plt.show()
        
def test_GP_samples_2d(plotFlag = True):
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5,2)
    obs = np.random.randn(5 ** 2)
    GP = GaussianProcess(design_pts, obs)
    grid = GaussianProcess.create_uniform_grid(-2,2,50,2)
    Z = GP.GP_at_points(grid, num_evals=1).T

    if plotFlag:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        X = grid[:,0]
        Y = grid[:,1]
        Z = Z.flatten()
        #Plot the surface
        ax.plot_trisurf(X, Y, Z)
        #Plot the design points
        ax.scatter(design_pts[:,0], design_pts[:,1], obs, color='green')
        plt.title('test_GP_samples_2d')
        plt.show()
        
def test_check_mem():
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5,2)
    obs = np.random.randn(5 ** 2)
    GP = GaussianProcess(design_pts, obs, total_calls = 1)
    GP.check_mem()
    assert len(GP.X) == 2 * (5 ** 2)
    assert len(GP.Y) == 2 * (5 ** 2)
    
def test_GP_GP_bridge_1d():
    design_pts = GaussianProcess.create_uniform_grid(-2,2,5)
    obs = np.random.normal(size = 5)
    GP = GaussianProcess(design_pts, obs)
    
    plt.figure()
    for _ in range(1):
        grid = np.random.uniform(low = -2, high = 2 , size = (3000,1))
#        OR uniform grid in random order:
#        grid = GaussianProcess.create_uniform_grid(-2,2,3000,1)
#        np.random.shuffle(grid)
        
        for x in grid:
            GP.GP_eval(x)
           
        #Sort to plot nicer
        X,Y = GP.get_data()
        ind = np.argsort(X.flatten())
        X = X.flatten()[ind]
        Y = Y[ind]
        #Plot results:
        plt.plot(X, Y)
        GP.reset()
    plt.plot(design_pts, obs, 'ro')
    plt.title('test_GP_GP_bridge_1d')
    plt.show()    
    
def test_GP_GP_bridge_2d():
    design_pts = GaussianProcess.create_uniform_grid(-2, 2, 5, 2)
    obs = np.random.randn(5 ** 2)
    GP = GaussianProcess(design_pts, obs)
    
    #Plot results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    grid = GaussianProcess.create_uniform_grid(-2, 2, 50, 2)
    np.random.shuffle(grid)
    
    for x in grid:
        GP.GP_eval(x[np.newaxis, :])
        
    points, Z = GP.get_data()
    X = points[:,0]
    Y = points[:,1]
    #Plot the surface
    ax.plot_trisurf(X, Y, Z)
    #Plot the design points
    ax.scatter(design_pts[:,0], design_pts[:,1], obs, color='green')
    plt.title('test_GP_GP_bridge_2d')
    plt.show()
    
def test_find_and_add_2d():
    """Green are closen points, red is x, blue dotted are X (not chosen)"""
    X = np.random.uniform(size=(20,2))
    GP = GaussianProcess(X, np.ones(X.shape[0]))
    
    #add some random data
    for _ in range(10):
        GP.GP_eval(np.random.uniform(size=(1,2)))
    
    x = np.random.uniform(size=(1,2))
    ind, points = GP.find_closest(x)
    
    plt.figure()
    plt.plot(X[:,0], X[:,1],'bo')
    plt.plot(x[:,0], x[:,1],'ro')

    plt.plot(points[:,0], points[:,1], 'go')
    plt.title('test_find_closest_2d')
    plt.show()
    
def test_find_and_add_1d():
    """Green are closen points, red is x, blue dotted are X (not chosen)"""
    X = np.arange(10)/10
    GP = GaussianProcess(X, np.ones(X.shape))
    
    #add some 'bad' data
    GP.GP_eval(np.array([-0.1]))
    GP.GP_eval(np.array([1]))
    #add some random data
    for _ in range(15):
        GP.GP_eval(np.random.uniform(size=(1)))
        
    x = 1.2*np.random.uniform(size=(1))
    ind, points = GP.find_closest(x)
    
    points = np.squeeze(points, axis=1)
    X, _ = GP.get_data()
    
    #make sure data is sorted
    assert np.all(X[:-1] <= X[1:])
    
    plt.figure()
    plt.hlines(1,0,1)
    plt.eventplot(X, orientation='horizontal', colors='b', linestyles='dotted')
    plt.eventplot(x, orientation='horizontal', colors='r')
    plt.eventplot(points, orientation='horizontal', colors='g', linestyles='dashed')
    plt.axis('off')
    
    plt.title('test_find_closest_1d')
    plt.show()
    
        
if __name__ == '__main__':
#    test_r2_distance()
#    test_GP_bridge()
#    test_check_mem()
    
#    np.random.seed(100)  
#    test_GP_samples_1d()
#    np.random.seed(100)
#    test_GP_GP_bridge_1d()
#    np.random.seed(100)
#    test_GP_interp_1d()
    
#    np.random.seed(100)
#    test_GP_samples_2d()
#    np.random.seed(100)
#    test_GP_GP_bridge_2d()
#    np.random.seed(100)
#    test_GP_interp_2d()
    
    test_find_and_add_1d()
#    test_find_and_add_2d()