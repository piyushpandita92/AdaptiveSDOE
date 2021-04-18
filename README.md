# AdaptiveSDOE
This module is called AdaptiveSDOE for Black-Box
Bayesian Optimal Design of Experiments 
 
It needs the support of the following PYTHON packages.
1. pyDOE 
2. GPy (version 1.9.2, mandatory)
3. matplotlib(version 2.0.0, best vizualization)
4. seaborn (version 0.7.1, best vizualization)
5. gpflow (clones from source, version 0.4.0)
6. tensorflow 

To install the package do the following:
pip install git+git://github.com/piyushpandita92/beast.git  

or clone the repository and run ```python setup.py install```.

Import the package like as follows:
 ```import beast```

The simple examples ex1.py, ex2.py provide a self explanatory overview of using BEAST.
This code works for estimating/inferring statistics, that are operators on the black-box function (so the user would have to include that in their function object).

The user mainly needs to specify the objective function ```obj_func``` as an object, number of iterations (samples to be collected depending on the budget) ```max_it```, number of designs of the discretized input space (for calculating the value of the EKLD criterion) ```X_design```. 
The user needs to supply a ```qoi_func``` function which can take an array of samples of the function f and apply the required transformation to return the corresponding values of the QoI. Examples of some pre-defined ```qoi_func```s have been provided in ```examples/useful_functions.py```.

Note: The methodology should be used with the inputs transformed to [0, 1]^{d} cube and outputs roughly normalized to a standard normal.

The index for a particular QoI has been pre-defined for the following 5 QoIs:
1. statistical expectation
2. statistical variance
3. maximum value of the function
4. minimum value of the function
5. 2.5th percentile of the function

One may define a new QoI by appending a function  similar to the ones already defined in the file ```examples/useful_functions.py```.

For sequential design  (one suggested design/experiment at a time):
Running the code: the examples in the ```examples``` directory can be called from the command line with a set of arguments as follows: python examples/ex1.py 1. This will infer the statistical expectation of the function.

After each iteration a plot depicting the state of the function is generated for 1d problems, this can be controlled by a ```plots``` flag set to 0 or 1.  


More documentation to follow.
If you have any questions about pydes, contact me at piyush100078@gmail.com .
