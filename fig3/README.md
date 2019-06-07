# Code to generate Figure 3

## Plotting performance data

A cached version of the performance data is stored in the MultiAlternativeDecisions/shared/optimParams_paper folder. If you change the folder structure on your system, make sure that you change the path in the plot file below. To plot this data, use
```
plot_relativeModelPerformance()    % add argument 'v' for value-based (default), 'p' for perceptual
```
at the MATLAB command line. 

## Re-generating performance data (not recommended)

To re-generate the performance data, call
```
optim_AllModels
```
at the MATLAB command line. Please beware that this script takes a very (very) long time to run. Note that this depends on optimParams.m, which is also present in the folder and which you could use to optimise the parameters of any model with a specified number of options and objective.
