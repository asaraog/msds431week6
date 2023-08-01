# Concurrency and Machine Learning

## Project Summary

This project aims to evaluate Go's concurrent programming framework for training and testing machine learning models namely linear regression. In particular, this project uses the [Boston Housing Study (1970)](http://lib.stat.cmu.edu/datasets/boston), commonly used by statisticians to predict housing prices by others (Brownlee 2020). This dataset was modified by others (Miller 1999) to remove the feature 'B' encoding racial segregation.

Machine learning models will utilize the [gonum library](https://pkg.go.dev/gonum.org/v1/gonum). The Go implementation will test linear regression models with varying regulization and/or concurrency. The data will be split approximately 80/20 with 80% (400/506) observations used for training and 20% (106/506) of the data for testing. RMSE and R<sup>2</sup> values are reported for each model tested with alpha=0.000001 (close to no regularization) and alpha=0.51 (optimized regularization level by Brownlee 2020).

All models are benchmarked for runtime using 'time' before commands in the command line and the concurrency flag 0 or 1. Concurrency significantly reduced runtime with runtimes of 0.016s and 0.042s with and without regularization respectively. Models are trained 100 times. The models with alpha=0.000001 performed better than with alpha = 0.51. Note that the data is resampled every time the program is run and results in differing results when run. In one run, the RMSE and R<sup>2</sup> was 61.5, 0.625 and 51.5, 0.737 for models with and without regularization respectively showing lower RMSE and higher R<sup>2</sup> score. 


## Files

*saraogeeweek6.go:* \
Main routine loads boston.csv file and uses flag '-concurrency' to select OFF (0) or ON (1) as an integer.

*saraogeeweek6_test.go:* \
Unit test for regression ensuring dimensionality. It also tests if R<sup>2</sup> values are within reasonalbe limits of 0 and 1.

*boston.csv* \
Input file for Boston housing data from Miller 1999.

*Week6* \
Unix executable file of cross-compiled Go code for Mac/Windows. 


## Installation

Download or git clone this project onto local machine into folder on local machine.
```
git clone https://github.com/asaraog/msds431week6.git
cd msds431week6
time ./Week6 -concurrency 0
time ./Week6 -concurrency 1

```


## References

Brownlee, Jason. 2020. “How to Develop Ridge Regression Models in Python.” MachineLearningMastery.Com (blog). October 8, 2020. https://machinelearningmastery.com/ridge-regression-with-python/.

Miller, Thomas W. 1999. "The Boston splits: Sample size requirements for modern regression." 1999 Proceedings of the Statistical Computing Section of the American Statistical Association, 210–215.

