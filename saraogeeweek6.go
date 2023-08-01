package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func main() {

	//Reads in each line of input as an array of strings and into [][]float64
	file, _ := os.Open("boston.csv")
	reader := csv.NewReader(file)
	record, _ := reader.ReadAll()
	//headers := records[1][:]
	//removes first column of categorical variable. I did try adding it by one hot encoding but I had poor results
	records := record[1:][:]

	Xrand, Yrand := StrMatrix(records) //takes in records of strings and turns into mat.Dense type

	//Splits X matrix into training and test
	Xtrain := Xrand.Slice(0, 400, 0, 14)
	Xtest := Xrand.Slice(400, 506, 0, 14)
	Ytrain := Yrand.Slice(0, 400, 0, 1)
	Ytest := Yrand.Slice(400, 506, 0, 1)

	concurrency := 0
	flag.IntVar(&concurrency, "concurrency", 0, "True/False for concurrency")
	flag.Parse()

	switch concurrency {
	case 0:
		var rmse1 float64
		var r21 float64
		var rmse2 float64
		var r22 float64
		for g := 0; g < 100; g++ {
			Beta1 := RegularizedLinearRegression(0.000001, Xtrain, Ytrain) //close to no regularization
			Beta2 := RegularizedLinearRegression(0.51, Xtrain, Ytrain)     //regularization
			rmse1 = RMSE(Beta1, Xtest, Ytest)
			r21 = R2(Beta1, Xtest, Ytest)
			rmse2 = RMSE(Beta2, Xtest, Ytest)
			r22 = R2(Beta2, Xtest, Ytest)
		}
		fmt.Printf("RMSE for Linear Regression: %f\n", rmse1)
		fmt.Printf("R2 for Linear Regression: %f\n", r21)
		fmt.Printf("RMSE for Regularized Linear Regression: %f\n", rmse2)
		fmt.Printf("R2 for Regularized Linear Regression: %f\n", r22)

	case 1:
		rmsech1 := make(chan float64)
		r2ch1 := make(chan float64)
		rmsech2 := make(chan float64)
		r2ch2 := make(chan float64)
		for g := 0; g < 100; g++ {
			go func() {
				Beta1 := RegularizedLinearRegression(0.000001, Xtrain, Ytrain) //close to no regularization
				rmsech1 <- RMSE(Beta1, Xtest, Ytest)
				r2ch1 <- R2(Beta1, Xtest, Ytest)
				Beta2 := RegularizedLinearRegression(0.51, Xtrain, Ytrain) //regularization
				rmsech2 <- RMSE(Beta2, Xtest, Ytest)
				r2ch2 <- R2(Beta2, Xtest, Ytest)
			}()
		}
		rmse1 := <-rmsech1
		r21 := <-r2ch1
		fmt.Printf("RMSE: %f\n", rmse1)
		fmt.Printf("R2: %f\n", r21)
		rmse2 := <-rmsech2
		r22 := <-r2ch2
		fmt.Printf("RMSE: %f\n", rmse2)
		fmt.Printf("R2: %f\n", r22)
	}
}

func StrMatrix(records [][]string) (X, Y mat.Dense) {
	//takes in records of strings and turns into mat.Dense type
	input_matrix := make([][]float64, len(records))
	for i := range records {
		input_matrix[i] = make([]float64, len(records[0]))
		for y := range records[i] {
			input_matrix[i][y], _ = strconv.ParseFloat(records[i][y], 64)
		}
	}

	//Makes Y matrix from mv variable (last column from input matrix)
	Ymat := mat.NewDense(len(records), 1, nil)
	for m := range records {
		Ymat.Set(m, 0, input_matrix[m][len(input_matrix[0])-1])
	}

	//Makes X matrix
	Xmat := mat.NewDense(len(records), len(input_matrix[0]), nil)
	for n := 0; n < 506; n++ {
		//sets Beta0 or constant term
		Xmat.Set(n, 0, 1)
		for p := 0; p < 13; p++ {
			Xmat.Set(n, p+1, input_matrix[n][p])
		}
	}

	//randomization of Xmat rows or shuffling of data
	a := rand.Perm(506) //outputs random indices for rows
	Xrand := mat.NewDense(len(records), 14, nil)
	Yrand := mat.NewDense(len(records), 1, nil)
	for t := 0; t < 506; t++ {
		Yrand.Set(t, 0, Ymat.At(a[t], 0))
		for u := 0; u < 14; u++ {
			Xrand.Set(t, u, Xmat.At(a[t], u))
		}
	}

	return *Xrand, *Yrand
}

func RegularizedLinearRegression(alpscalar float64, Xtrain, Ytrain mat.Matrix) (Beta mat.Dense) {
	//Initialization of variables
	var xx mat.Dense      //XtX
	var ixxi0 mat.Dense   //(XtX +alpha*I)^-1)
	var xy mat.Dense      // (XtY)
	var i0 mat.Dense      //I
	var alphai0 mat.Dense // alpha*I
	var xxi0 mat.Dense    //(XtX +alpha*I)

	// Makes a matrix of alphas because gonum does not support multiplication of matrices with scalars
	alpha := mat.NewDense(14, 14, nil)
	for i := 0; i < 14; i++ {
		for j := 0; j < 14; j++ {
			alpha.Set(i, j, alpscalar)
		}
	}
	alpha.Set(0, 0, 0) //Makes sure we do not regularize our first term
	//Training with Beta= ((XtX +alpha*I)^-1)*(XtY)
	xx.Mul(Xtrain.T(), Xtrain) //XtX
	i0.Pow(&xx, 0)             //I

	alphai0.Mul(&i0, alpha)    //(alpha*I)
	xxi0.Add(&xx, &alphai0)    //(XtX +alpha*I)
	ixxi0.Inverse(&xxi0)       //(XtX +alpha*I)^-1)
	xy.Mul(Xtrain.T(), Ytrain) // (XtY)
	Beta.Mul(&ixxi0, &xy)      //Beta
	return Beta
}

func R2(Beta mat.Dense, Xtest, Ytest mat.Matrix) (r2 float64) {
	//Initialization of variables
	var Ypred mat.Dense
	var e mat.Dense       //error or ytest-ypred
	var e2 mat.Dense      //error squared for r2 calculation
	var yYmean mat.Dense  //ytest-ymean for r2 calculation
	var yYmean2 mat.Dense //(ytest-ymean)^2 for r2 calculation

	//Testing and predictions
	Ypred.Mul(Xtest, &Beta)

	e.Sub(Ytest, &Ypred) //error or ytest-ypred
	e2.Mul(e.T(), &e)    //error squared for r2 calculation (SSR)

	//Creating Ymean matrix for ytest-ymean
	params, _ := Ytest.Dims()                //for denominator in mean calculation
	Yave := mat.Sum(Ytest) / float64(params) //calculates ymean
	Ymean := mat.NewDense(params, 1, nil)
	for i := 0; i < params; i++ {
		Ymean.Set(i, 0, float64(Yave))
	}

	//Calculates R^2 as 1-SSR/SSE
	yYmean.Sub(Ytest, Ymean)         //ytest-ymean for r2 calculation (SSE)
	yYmean2.Mul(yYmean.T(), &yYmean) //(ytest-ymean)^2 for r2 calculation (SSE)
	r2 = 1 - e2.At(0, 0)/yYmean2.At(0, 0)
	return r2

}

func RMSE(Beta mat.Dense, Xtest, Ytest mat.Matrix) (rmse float64) {
	//Initialization of variables
	var Ypred mat.Dense
	var e mat.Dense  //error or ytest-ypred
	var e2 mat.Dense //error squared for r2 calculation

	//Testing and predictions
	Ypred.Mul(Xtest, &Beta)

	e.Sub(Ytest, &Ypred) //error or ytest-ypred
	e2.Mul(e.T(), &e)    //error squared for r2 calculation
	rmse = e.Norm(2)     //Calculates RMSE
	return rmse

}
