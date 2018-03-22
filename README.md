# Problem Statement

Implement a small library in C++ using expression templates and modern C++11/14/17 features like generic lambdas to:

represent a matrix of numerical types (int, long, float, double, complex,....)
compute algebraic expressions including + and *, += and *=. Matrix multiplication can be implemented with the most simple algorithm
fit in one header file
provides a test program which can measure its speed of execution for each examples provided






# Solution
- `matrix.h` : - header file whixh consist of operations related to matrix using generic lamdas.
- `timer.h` :-  header file for giving the execution time of the process.
- `main.cpp` :- for testing the above header files.

## Requirements

- compiler supporting the C++17 standard(I use clion for testing) 


## Template class `cmat::matrix`

- has template parameters `<E,M,N,S>` where `E` is the element type, `S`, is the storage format, `M` and `N` the number of rows and columns of the matrix
- stores elements using `std::vector<E>` either in column-major or row-major storage format

## Template class `cmat::timer`

- encapsulates `std::chrono::high_resolution_clock` for convenient runtime measurements.
- has one template parameter `D` which can be e.g. of type `cmat::nanoseconds`.



## Overloaded operators

- elementwise arithmetic operators `+`,`-`,`*`,`/` with scalars and matrices
- elementwise assignment operators `+=`,`-=`,`*=`,`/=` with scalars and matrices
- matrix-multiplication operator `|`
- matrix-transposition operator `!`

