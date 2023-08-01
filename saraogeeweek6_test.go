package main

import (
	"encoding/csv"
	"os"
	"testing"
)

func TestRegression(t *testing.T) {

	//Reads in Data
	file, _ := os.Open("boston.csv")
	reader := csv.NewReader(file)
	record, _ := reader.ReadAll()
	records := record[1:][:]

	//Testing of StrMatrix to ensure correct number of parameters and data Xmn-(506X14) Ymn-(506X1)
	Xrand, Yrand := StrMatrix(records) //takes in records of strings and turns into mat.Dense type
	rowsX, colsX := Xrand.Dims()
	rowsY, colsY := Yrand.Dims()
	if rowsX != 506 {
		t.Errorf("Error in Regularized Linear Regression function output Beta. Rows in wrong dimension")
	}
	if colsX != 14 {
		t.Errorf("Error in Regularized Linear Regression function output Beta. Columns in wrong dimenstion.")
	}
	if rowsY != 506 {
		t.Errorf("Error in Regularized Linear Regression function output Beta. Rows in wrong dimension")
	}
	if colsY != 1 {
		t.Errorf("Error in Regularized Linear Regression function output Beta. Columns in wrong dimenstion.")
	}

	//Splits X matrix into training and test
	Xtrain := Xrand.Slice(0, 400, 0, 14)
	Xtest := Xrand.Slice(400, 506, 0, 14)
	Ytrain := Yrand.Slice(0, 400, 0, 1)
	Ytest := Yrand.Slice(400, 506, 0, 1)

	//Testing for LinearRegression function to ensure dimenionality as 14X1
	alpscalar := 0.000001
	Beta := RegularizedLinearRegression(alpscalar, Xtrain, Ytrain)
	rows, cols := Beta.Dims()
	if rows != 14 {
		t.Errorf("Error in Regularized Linear Regression function output Beta. Rows in wrong dimension")
	}
	if cols != 1 {
		t.Errorf("Error in Regularized Linear Regression function output Beta. Columns in wrong dimenstion.")
	}

	//Testing of R2 function to ensure reasonable values, within constraints of between 0 and 1
	r2 := R2(Beta, Xtest, Ytest)
	lowerbound := r2 < 0
	upperbound := r2 > 1
	if lowerbound || upperbound {
		t.Errorf("R2 value is out of bounds.")
	}
}
