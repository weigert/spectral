# spectral

C++ Header-Only Method of Weighted Residuals for n-D PDEs

## Description

This library allows you to numerically solve problems stated as [linear operator + inhomogeneity] acting on a function in some domain. Solutions are represented as a linear combination of a basis set of functions and computed using the method of weighted residuals.

This is explained in more detail below.

### Method of Weighted Residuals

The method of weighted lets you numerically solve problems of the form `P[f] = 0`, where `f` is a function in some domain `D` and `P` is an operator.

Defining the operator `P[f] = Lf - r` (linear + inhomogeneity), we try to find an approximate solution `f_N`.

We express `f_N` as a weighted sum of `K` basis functions `f_k` with an imposed inhomogeneity `f_R`:

    f_N = f_R + sum(k < K, w_k*f_k)

where `w_k` are the weights. The "residual" `R` is the given by `P[f_N] = R`.

Integrating the residual `R` over the domain `D` and finding the weights `w_k` which minimize the expression results in a linear system which can be solved for the weights.

Note that the number of basis functions `K` and the basis functions themselves are chosen in advance.

**This library allows you to choose `P`, `D`, `f_k` and `K` to solve for `w_k`.**

### Applications

Some example applications are:

- Fit data points in n-dimensions to some basis function representation
- Solve differential equations
- Least squares energy minimization for complex formulated problems
- Perform a discrete fourier / cosine transform on a non-regular grid
- Interpolate data

## features

### Residual Methods

The library currently implements three weighted residual solvers:

- Least Squares Method Weighted Residual
- Galerkin
- Collocation Method (Residuals Disappear at Sample Points)

These methods are expressed as a linear system and solved using Eigen3.

### Basis Function Sets

The library currently includes the following basis sets:

- complex fourier modes
- chebyshev polynomials
- regular polynomials
- cosine functions, i.e. cosine transform

## todo

- Arbitrary maps between R^n -> R^m, i.e. multidimensional problems
- Operator class for choosing the problem formulation
- Domain class for transforming coordinates / testing validity
- Better utilization of class inheritance to clean the code
- Mean squared error output
- Better templating

## other information

Data is currently being visualized using TinyEngine. This dependency will be separated out later.
