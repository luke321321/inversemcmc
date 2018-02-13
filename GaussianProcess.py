import numpy as np
import math
import numbers
from scipy.special import gamma, kv
from scipy.interpolate import RegularGridInterpolator, interp1d

class GaussianProcess:
    def __init__(self, points, data, total_calls = 1000, nu=1/2, sig2 = 1, lam = 1):
        """Initialises a Guassian Emulator with the kernel being a Matern kernel.
        Defaults: to exponential kernel
        Gaussian: nu = np.inf
        Exp: nu = 1/2
        """
        #tolerance for checking if something is close to 0
        self.tol = 1e-14

        self.points, self.data = self.reshape_points(points, data)
        
        self.data_length = len(data)
        
        self.nu = nu
        self.sig2 = sig2
        self.lam = lam
        self.dim = self.points.shape[1]
        self.total_calls = max(total_calls, self.points.shape[0])
        
        self.mean, self.kernel = self.create_matern()
         
        self.reset()
        
    @staticmethod
    def reshape_points(points, data):
        """if the points are 1d: sort them and append empty axis for r2_distance method
        Returns points and data after shaping and sorting"""
        if len(points.shape) == 1:
            index = np.argsort(points)
            return points[index, np.newaxis], data[index]
        else:
            return points, data
            
        
    def create_matern(self, data=None, points=None):
        """Creates a Guassian Emulator with the kernel being a Matern kernel.
        Returns functions mean(u), kernel_N(u,v)

        Note for functions mean(u), kernel_N(u,v)
        If u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector
        and the same for v
        """
        
        if data is None:
            data = self.data
        if points is None:
            points = self.points
        #make sure the points are shaped correctly - might not be needed
        points, data = self.reshape_points(points, data)
        
        if len(points) != 1:
            kernel_star_inverse = np.linalg.inv(self.k(points, points))
            kernel_star_inverse_dot_phi_star = kernel_star_inverse @ data

            mean = lambda u : self.k(u, points) @ kernel_star_inverse_dot_phi_star
            kernel = lambda u,v: self.k(u, v) - (self.k(v, points) @ kernel_star_inverse
                                @ self.k(u, points).T)
        #if only 1 point is given then kernel_star_inverse = 1 and objects are scalars
        else:
            mean = lambda u : self.k(u, points) * data
            kernel = lambda u,v: self.k(u, v) - self.k(v, points) * self.k(u, points).T
            
        return mean, kernel
    
    def k(self, u, v):
        r2 = GaussianProcess.r2_distance(u,v)

        if self.nu == np.inf:
            return np.exp(-r2)
        elif self.nu == 0.5:
            return self.sig2*np.exp(-np.sqrt(r2)/self.lam)
        else:
            rt2_nu = math.sqrt(2*self.nu)/self.lam
            const = (self.sig2/gamma(self.nu)*math.pow(2, self.nu-1))
            kuv = const*np.power(rt2_nu*r2, self.nu)*kv(self.nu, rt2_nu*r2)

            #if r2[i,j] = 0 then want kernelStar[i,j] = 1
            #Asymptotics when r2 = 0 not the best
            where_NaNs = np.isnan(kuv)
            kuv[where_NaNs] = 1
            return kuv

    def GP_at_points(self, points, num_evals=1):
        """Returns a Gaussian Process evalued at the grid points"""
        val = np.random.multivariate_normal(self.mean(points), self.kernel(points, points),
                                             num_evals)
        if num_evals > 1:
            val = val.T
        return val

    def GP_interp(self, num_grid_points, num_evals=1):
        """Returns a function that is a GP evaluated exactly at the grid points
        and uses linear intepolation to evaluate it at other points
        
        The returned function is really a vector of functions of size num_evaluations"""
        #TODO: only currently returns 1 evaluation of the GP need to call again for another eval
        if num_evals != 1:
            raise Exception('Not yet implemented')
        dp_min = np.amin(self.points)
        dp_max = np.amax(self.points)
        grid = self.create_uniform_grid(dp_min, dp_max, num_grid_points, self.dim)
        if self.dim == 1:
            return interp1d(grid.squeeze(), self.GP_at_points(grid).squeeze(), copy=False, assume_sorted=True)
        else:
            #reshape grid and data to fit the interpolator method
            grid_points_grid = [np.linspace(dp_min, dp_max, num_grid_points) for _ in range(self.dim)]
            GP_eval = self.GP_at_points(grid).reshape(num_grid_points, num_grid_points)
            return RegularGridInterpolator(grid_points_grid, GP_eval)
        

    def reset(self):
        #setup empty arrays to store data and coordinates in for Brownian bridge
        self.X = np.concatenate((self.points, 
                                np.zeros((self.total_calls - self.data_length, self.dim), dtype=self.points.dtype)))
        self.Y = np.concatenate((self.data, 
                                np.zeros(self.total_calls - self.data_length, dtype=self.data.dtype)))
        self.index = self.data_length
        
    def get_data(self):
        X = self.X[:self.index]
        Y = self.Y[:self.index]
        return X,Y
    
    def GP_eval(self, x):
        #first check lengeth of array X and Y and expand if necessary
        self.check_mem()
        
        """First find closest 2*d points to x
        Ideally create RTree but first do naive thing.
        Runs in O(n) time instead of O(log(n)) for RTree insertion and nearest neighbou
        Really only need smallest convex hull in self.X that contains x"""
        
        #only check distance up to computed values of X - about 1/2 of X is 0s
        dist = self.r2_distance(x, self.X[:self.index])           
        
        #if already computed this value before:
        ind_min_dist = np.argmin(dist)
        if dist[ind_min_dist] < self.tol:
            return self.Y[ind_min_dist]
        
        #indices of closest points to x in each dimension (up to 2*d points)
        ind = self.find_closest(x, self.X[:self.index], dist)
               
        points = self.X[ind]
        data = self.Y[ind]
        data_new = self.Brownian_bridge(x, points, data)
        
        #add new data to class
        self.X[self.index] = x
        self.Y[self.index] = data_new
        self.index += 1

        return data_new
    
    @staticmethod
    def find_closest(x, X, dist=None):
        """Finds a closest points of X to x in each signed direction
        Returns: index an array of lists st closest points are X[index] 
        
        This is a bottleneck and have just done the naive thing here"""
        
        #if dim = 1 reshape to reuse code
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
            x = x[:, np.newaxis]
        dim = X.shape[1]
        
        if dist is None:
            dist = GaussianProcess.r2_distance(x, X)
                   
        """Idea: In each 2*d axis direction find nearest point to x and choose it (by its index).
        If no point in that direction then doesn't matter"""
        
        diff = X - x
        #use set structure to easily ensure values are unique
        index = set()

        for d in range(dim):
            ind_pos = np.nonzero(diff[:, d] > 0)
            ind_neg = np.nonzero(diff[:, d] < 0)
            dist_pos = dist[ind_pos]
            dist_neg = dist[ind_neg]
            
            #try/except incase ind_pos is empty
            try:            
                closest_ind = np.argmin(dist_pos)
                index.add((ind_pos[0][closest_ind]))
            except:
                pass
            try:            
                closest_ind = np.argmin(dist_neg)
                index.add((ind_neg[0][closest_ind]))
            except:
                pass
        
        #returns the indices as a array of lists
        return np.array(list(index))
            
    def check_mem(self):
        if(self.index == len(self.X)):
            #double length of arrays
            self.X = np.concatenate((self.X, np.zeros((len(self.X), self.dim), dtype=self.X.dtype)))
            self.Y = np.concatenate((self.Y, np.zeros(len(self.Y), dtype=self.Y.dtype)))
    
    def Brownian_bridge(self, x, points, data):
        """Uses a small Gaussian process to evaluate the GP at x given it at points
        
        x: point to calculate the GP at
        Points: np array (x0, x1, ...)
        Data: np array (f(x0), f(x1), ...)"""
        
        mean, ker = self.create_matern(data = data, points = points)

        if len(x.shape) == 1:
            x = x.reshape(1, self.dim)
        all_pts = np.concatenate((x, points))
        val = np.random.multivariate_normal(mean(all_pts), ker(all_pts, all_pts), check_valid='ignore')
        return val[0]


    @staticmethod
    def create_uniform_grid(min_range, max_range, n, dim=1):
        """Creates a uniform grid with n points in each dimension on 
        [min_range, max_range]^dim"""
        
        if dim > 1:
            grid_points_grid = np.meshgrid(*[np.linspace(min_range, max_range, n) for _ in range(dim)])
            grid_points = np.hstack(grid_points_grid).swapaxes(0, 1).reshape(dim, -1).T
        else:
            grid_points = np.linspace(min_range, max_range, n)[:, np.newaxis]
        return grid_points
    
    @staticmethod
    def r2_distance(u, v):
        """Calculates the l^2 distance squared between u and v for u,v being points,
        vectors or list of vectors and returns a point, vector or matrix as appropriate.
        Dimensions of the vectors has to be the same

        Note if u.shape = (4,1) then u is an array of 1d points, if u.shape = (4,) then u is a vector.

        If u.shape = (m x d), v.shape = (n x d) then r2.shape = (m x n)
        If u.shape = (d,)     v.shape = (n x d) then r2.shape = (n)
        If u.shape = (m x d), v.shape = (d,)    then r2.shape = (m)
        If u.shape = (m,)     v.shape = (n,)    then r2.shape = (m x n)
        If u.shape = (d,)     v.shape = (d,)    then r2       = float
        If m or n = 1 then that dimension is squeezed out of r2.shape"""
                
        #First deal with floats sensibly and return
        if isinstance(u, numbers.Number) or isinstance(v, numbers.Number):
            r2 = np.squeeze(np.square(u-v))
            return r2

        #Now check dimensions and inputs
        dim_U = len(u.shape)
        dim_V = len(v.shape)

        #Can remove these asserts to speed up the function
        assert dim_U <= 2 and dim_V <= 2
        if dim_U == 2 and dim_V == 2:
            assert u.shape[1] == v.shape[1]
        elif dim_U == 1 and dim_V == 2:
            assert u.shape[0] == v.shape[1]
        elif dim_U == 2 and dim_V == 1:
            assert u.shape[1] == v.shape[0]
        
        if dim_U == 1 and dim_V == 1:
            #if 4th case append axes to get correct shape
            if u.shape[0] != v.shape[0]:
                V = v[np.newaxis,:]
                U = u[:,np.newaxis]
                r2 = np.squeeze(np.square(U - V))
            else:
                r2 = np.sum(np.square(u - v))
        elif dim_U < 2 or dim_V < 2:
            #cases 3 and 4, make u have the smallest dimension
            if dim_U > dim_V:
                u, v = v, u
            U = u[np.newaxis, :]
            r2 = np.sum(np.square(U - v), 1)
        else:
            #Deal with large u and v incase of out of memory errors
            try:
                U = u[:, np.newaxis, :]
                V = v[np.newaxis,...]
                r2 = np.squeeze(np.sum(np.square(U - V), 2))
            except MemoryError:
                #try to sum a slower way instead without creating a large intermediate array
                r2 = np.zeros((u.shape[0], v.shape[0]))
                if u.shape[0] < v.shape[0]:
                    for i in range(u.shape[0]):
                        r2[i,:] = np.sum(np.square(u[i][np.newaxis, :] - v), 1)
                else:
                    for j in range(v.shape[0]):
                        r2[:,j] = np.sum(np.square(u - v[j][np.newaxis, :]), 1)
        return r2