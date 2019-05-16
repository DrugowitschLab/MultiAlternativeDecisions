# Code to generate Figure S3

## Plotting performance data

A cached version of the performance data is stored in the different `.mat`-files. To plot this data, use
```
plotParamEffect
```
at the MATLAB command line.

## Re-generating performance data (not recommended)

To re-generate the performance data, call
```
computeParamEffect
```
at the MATLAB command line. Please beware that this script takes a very (very) long time to run.
