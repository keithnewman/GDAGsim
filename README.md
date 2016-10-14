# GDAGsim

## Java library for conditional simulation of Gaussian DAG models

### Version 0.5.10 (14 Oct 2016)

A Java version of the [GDAGsim](https://www.staff.ncl.ac.uk/d.j.wilkinson/software/gdagsim/) sofware originally written in C by [Professor Darren Wilkinson](https://www.staff.ncl.ac.uk/d.j.wilkinson/).  Includes a JUnit test class for the `GDAG` class.

Requires an installation of [Parallel Colt](https://sites.google.com/site/piotrwendykier/software/parallelcolt) version 0.9.4.  This GDAGsim repository comes with a modified version of the `SparseCholeskyDecompositionAlgorithm` Class from Parallel Colt, containing additional functions designed to help with the efficient processing of GDAG models, along with methods for quicker sampling and likelihood calculations.

Written for Java 1.6.

## Authors

[Keith Newman](mailto:keith.newman@ncl.ac.uk) (based on software produced by Darren J. Wilkinson).

`SparseCholeskyDecompositionAlgorithmExtended` is a modified version of the `SparseCholeskyDecompositionAlgorithm` class in Parallel Colt by Piotr Wendykier.

## Literature

Concepts used in this software are discussed in,

> Wilkinson, D. J. &amp; Yeung, S. K. H. (2004). [A sparse matrix approach to Bayesian computation in large linear models.](http://dx.doi.org/10.1016/S0167-9473(02)00252-9) *Computational Statistics and Data Analysis*, **44**(3):493&ndash;516.
