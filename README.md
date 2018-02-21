This code contains an example of using a Gaussian Process Approximation in Bayesian inverse problems.
In particular we use the Gaussian Process Emulator during a simple MCMC estimating the solution to a PDE.

We look at two different examples.

The first example, `MCMC_A.py` is from section 5 of this [paper](http://dx.doi.org/10.1090/mcom/3244) (also available on [arXiv](https://arxiv.org/abs/1603.02004)).

The second example is using applying a Gaussian Process Approximator to the problem from section 5 of this [paper](https://doi.org/10.1007/s11222-016-9671-0).
Here they have instead used a randomised PDE solver using random basis function for their FEM.
We randomise differently using a Gaussian process approximator.

Note because the code currently has a naive way of searching spatial data it runs in `O(d*n^2)`, where `n` is length of MCMC and `d` is the dimension of the parameter space.
Using an R* Tree (or another data structure) should reduce this down to `O(d*n*log(n))`.

### How do I get set up? ###

Required dependencies:

	numpy, scipy, matplotlib

Optional dependencies:

	hypothesis and tqdm

If you don't have the optional dependencies installed then the code can be simply modified as follows.

`tqdm` is only used to give a progress bar for the MCMC run contained in `MCMC.py`.
To remove this dependency just replace the line (currently `line 31`) in `MCMC.py`

```python
	for i in tqdm(range(1,burn_time + length)):
```
by
```python
	for i in range(1,burn_time + length):
```
`hypothesis` is only required for running the test code (`test_GaussianProcess.py`).
	

### How do I run the code ###
Simply run `python MCMC_A.py` or `python MCMC_CGSSZ.py`
See the above file for different plotting and posterior distribution options.

### Testing ####
The tests are contained in `test_GaussianProcess.py`.
Since we're dealing with randomness most of the tests are graphical.
Simply uncomment the required tests at the bottom of the file.