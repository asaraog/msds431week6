# Concurrency and Machine Learning

## Project Summary

This project aims to evaluate Go's concurrent programming framework for training and testing machine learning models namely linear regression. In particular, this project uses the [Boston Housing Study (1970)](http://lib.stat.cmu.edu/datasets/boston), commonly used by statisticians to predict housing prices. This dataset was modified by others (Miller 1999) to remove the feature 'B' encoding racial segregation.

Machine learning models will utilize the [gonum library](https://pkg.go.dev/gonum.org/v1/gonum).The Go implementation will test k-fold crossvalidated linear regression models with and without regulization and/or concurrency. 

1. Linear regression
2. Linear regression with regularization
3. Linear regression with concurrent programming
4. Linear regression with regularization and concurrent programming

All models are benchmarked for runtime using 'time' before commands in the command line. We also see that models with concurrent programming had the same R ^2^ values in both models. Concurrency significantly reduced runtime by XXXs and XXXs for models with and without regularization respectively as shown in Table 1 below.

| Model  | R ^2^   | Runtime | 
| ------ | ------- | ------- | 
| 1      |   0.78  |    s    |
| 2      |   0.78  |    s    |
| 3      |   0.78  |    s    |
| 4      |   0.80  |    s    |


## Installation

Download or git clone this project onto local machine into folder on local machine.
```
git clone https://github.com/asaraog/msds431week6.git
cd msds431week6
time ./model1
time ./model2
time ./model3
time ./model4

```

## References

Miller, Thomas W. 1999. "The Boston splits: Sample size requirements for modern regression." 1999 Proceedings of the Statistical Computing Section of the American Statistical Association, 210–215.

