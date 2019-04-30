# mexFiles

Mex code for faster simulation of simple version of the network model. The code relies on the [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/)  for random number generation, and was tested with version 2.3 of that library.

It has only been tested on OS X, where is can be compiled by running
```
mex -I/usr/local/include -lgsl simulateDiffusionRTC.c
```
on the MATLAB command prompt. Here `/usr/local/include` is the assumed location of the GSL headers.