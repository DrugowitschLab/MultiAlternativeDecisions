# Code to generate Figure 5

## Plotting activity normalization

To plot the activity normalization (Figure 5a) of our full model, use
```
activityNormalization
```
at the MATLAB command line. Note that we have obtained the data part of Figure 5a (left) from Figure 1e of Louie et al., Reward Value-Based Gain Control: Divisive Normalization in Parietal Cortex, *PNAS* (2011). We extracted the data points using [Web-Plot-Digitizer](https://automeris.io/WebPlotDigitizer/) and use them to plot our own version: `fig5_data.m`.

## Plotting IIA violation

We achieve the IIA violation (Figure 5b) by adding noise to the accumulator, or to the decision bounds. To plot the IIA violation, use 
```
iiaViolation_noisyBound
```
or 
```
iiaViolation_noisyAccumulator
``` 
at the MATLAB command line.

## Plotting performance data of the model with noise
A cached version of the performance data is stored in the MultiAlternativeDecisions/shared/optimParamsNoisy folder. If you change the folder structure on your system, make sure that you change the path in the plot file. To plot this data, use
```
plot_noisyModelPerformance
```
at the MATLAB command line. 

To regenerate performance data of the noisy model (not recommended), use `optim_AllModels.m` from the folder MultiAlternativeDecisions/fig3/, and while changing `noise=0` to `noise=1` in that script. Please beware that this script takes a very (very) long time to run, even longer than the non-noisy version.
