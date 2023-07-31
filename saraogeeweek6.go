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
	records := record[1:][:]
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

	//Splits X matrix into training and test
	Xtrain := Xrand.Slice(0, 400, 0, 14)
	Xtest := Xrand.Slice(400, 506, 0, 14)
	Ytrain := Yrand.Slice(0, 400, 0, 1)
	Ytest := Yrand.Slice(400, 506, 0, 1)

	model := 1
	flag.IntVar(&model, "model", 1, "model number")
	flag.Parse()

	switch model {
	case 1:
		var rmse float64
		var r2 float64
		avgBeta := mat.NewDense(14, 1, nil)
		for g := 0; g < 100; g++ {
			var Beta mat.Dense
			Beta1000 := mat.NewDense(14, 1000, nil)

			// var valrmse []float64
			// var valr2 []float64
			for h := 0; h < 1000; h++ {
				// a := rand.Perm(400) //outputs random indices for rows
				// Xrand := mat.NewDense(400, 14, nil)
				// Yrand := mat.NewDense(400, 1, nil)
				// for t := 0; t < 400; t++ {
				// 	Yrand.Set(t, 0, Ytrain.At(a[t], 0))
				// 	for u := 0; u < 14; u++ {
				// 		Xrand.Set(t, u, Xtrain.At(a[t], u))
				// 	}
				// }
				// Xvaltrain := Xrand.Slice(0, 400, 0, 14)
				// Yvaltrain := Yrand.Slice(0, 400, 0, 1)
				Beta = RegularizedLinearRegression(3.8, Xtrain, Ytrain)

				for a := 0; a < 14; a++ {
					Beta1000.Set(a, h, Beta.At(a, 0))
				}

			}
			for j := 0; j < 14; j++ {
				sumBeta := mat.Sum(Beta1000.ColView(j)) / 1000
				avgBeta.Set(j, 0, sumBeta)
			}

			rmse, r2 = RegressionMetrics(*avgBeta, Xtest, Ytest)
			fmt.Println(mat.Sum(Beta1000.ColView(0)))
		}

		fmt.Println(rmse)
		fmt.Println(r2)
	case 2:
		var rmse float64
		var r2 float64
		for g := 0; g < 100; g++ {
			Beta := RegularizedLinearRegression(38, Xtrain, Ytrain)
			rmse, r2 = RegressionMetrics(Beta, Xtest, Ytest)
		}
		fmt.Println(rmse)
		fmt.Println(r2)
		// case 3:
		// Beta := RegularizedLinearRegression(38, Xtrain, Ytrain)
		// rmse, r2 := RegressionMetrics(&Beta, Xtest, Ytest)
		// fmt.Println(rmse)
		// fmt.Println(r2)

	}
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

func RegressionMetrics(Beta mat.Dense, Xtest, Ytest mat.Matrix) (rmse, r2 float64) {
	//Initialization of variables
	var Ypred mat.Dense
	var e mat.Dense       //error or ytest-ypred
	var e2 mat.Dense      //error squared for r2 calculation
	var yYmean mat.Dense  //ytest-ymean for r2 calculation
	var yYmean2 mat.Dense //(ytest-ymean)^2 for r2 calculation

	//Testing and predictions
	Ypred.Mul(Xtest, &Beta)

	e.Sub(Ytest, &Ypred) //error or ytest-ypred
	e2.Mul(e.T(), &e)    //error squared for r2 calculation
	params, _ := Ytest.Dims()
	Yave := mat.Sum(Ytest) / float64(params)
	Ymean := mat.NewDense(params, 1, nil)
	for i := 0; i < params; i++ {
		Ymean.Set(i, 0, float64(Yave))
	}
	yYmean.Sub(Ytest, Ymean)         //ytest-ymean for r2 calculation
	yYmean2.Mul(yYmean.T(), &yYmean) //(ytest-ymean)^2 for r2 calculation

	r2 = 1 - e2.At(0, 0)/yYmean2.At(0, 0) //Calculates R^2
	rmse = e.Norm(2)                      //Calculates RMSE
	return

}
