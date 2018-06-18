This code contains an example of using a Gaussian Process Approximation in Bayesian inverse problems.
In particular we use the Gaussian Process Emulator during a simple MCMC estimating the solution to a PDE.

We look at two different examples.

The first example, `MCMC_A.py` is from section 5 of this [paper](http://dx.doi.org/10.1090/mcom/3244) (also available on [arXiv](https://arxiv.org/abs/1603.02004)).

The second example, `MCMC_CGSSZ.py` is using applying a Gaussian Process Approximator to the problem from section 5 of this [paper](https://doi.org/10.1007/s11222-016-9671-0).
Here they have instead used a randomised PDE solver using random basis function for their FEM.
We randomise differently using a Gaussian process approximator.

Note because the code currently has a naive way of searching spatial data it runs in `O(n^3)`, where `n` is length of MCMC run.
Using an R* Tree (or another data structure) should reduce this down to `O(n*log(n))`.

### How do I get set up? ###

Required dependencies:

	numpy, scipy, matplotlib

Optional dependencies:

	hypothesis and tqdm

If you don't have the optional dependencies installed then `MCMC_A.py` and `MCMC_CGSSZ.py` will run fine but there won't be a progress bar.

`tqdm` is used to give a progress bar for the MCMC run contained in `MCMC.py` but the code will adapt if the `tqdm` is not installed.

`hypothesis` is only required for running the test code (`test_GaussianProcess.py`).
	

### How do I run the code ###
Simply run `python MCMC_A.py` or `python MCMC_CGSSZ.py`
See the above file for different plotting and posterior distribution options.

### Testing ####
The tests are contained in `test_GaussianProcess.py`.
Since we're dealing with randomness most of the tests are graphical.
Simply uncomment the required tests at the bottom of the file.