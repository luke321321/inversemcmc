This code contains an example of using a Gaussian Process Approximation in Bayesian inverse problems.
In particular we use the Gaussian Process Emulator during a simple MCMC estimating the solution to a PDE.
This example is modified from section 5 of this [paper](http://dx.doi.org/10.1090/mcom/3244) (also available on [arXiv](https://arxiv.org/abs/1603.02004)).
I modified boundary conditions to make the solution map 'more one-to-one'.

### How do I get set up? ###

Required dependencies:

	numpy, scipy, matplotlib

Optional dependencies:

	hypothesis and tqdm

If you don't have the optional dependencies installed then the code can be simply modified as follows.

`tqdm` is only used to give a progress bar for the MCMC run in `MCMC_A.py`.
To remove this dependency just replace the line (currently `48`) in `MCMC_A.py`

```python
	for i in tqdm(range(1,burn_time + length)):
```
by
```python
	for i in range(1,burn_time + length):
```
If you're not running the test code (`test_GaussianProcess.py`) then `hpyothesis` is not required.
	

### How do I run the code ###
Simply run `python MCMC_A.py`.
See the above file for different plotting and posterior distribution options.

### Testing ####
The tests are contained in `test_GaussianProcess.py`.
Since we're dealing with randomness most of the tests are graphical.
Simply uncomment the required tests at the bottom of the file.