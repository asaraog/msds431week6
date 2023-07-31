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

// From https://gosamples.dev/remove-duplicates-slice/
func unique(s []string) []string {
	inResult := make(map[string]bool)
	var result []string
	for _, str := range s {
		if _, ok := inResult[str]; !ok {
			inResult[str] = true
			result = append(result, str)
		}
	}
	return result
}

func main() {
	model := 1
	flag.IntVar(&model, "model", 1, "model number")

	flag.Parse()
	//Reads in each line of input as an array of strings
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
	//One hot encoding of categorical variable
	neighborhood := make([]string, len(records))
	for j := range records {
		neighborhood[j] = records[j][0]
	}
	neighborhoods := unique(neighborhood)
	onehot_neighborhood := make([][]float64, len(neighborhood))
	for z := range onehot_neighborhood {
		onehot_neighborhood[z] = make([]float64, len(neighborhoods))
	}

	for k := range neighborhood {
		for l := range neighborhoods {
			if neighborhood[k] == neighborhoods[l] {
				onehot_neighborhood[k][l] = 1
			}
		}
	}

	//Splits X matrix into training and test
	Xmat := mat.NewDense(len(records), 1+len(input_matrix[0])-1+len(onehot_neighborhood[0]), nil)
	for n := range records {
		//sets Beta0 or constant term
		Xmat.Set(n, 1, 1)

		//sets neighborhood and other matrix values
		for o := range onehot_neighborhood[0] {
			Xmat.Set(n, o+1, onehot_neighborhood[n][o])
		}
		for p := range input_matrix[0][:len(input_matrix[0])-1] {
			Xmat.Set(n, p+len(onehot_neighborhood[0])+1, input_matrix[n][p])
		}
	}

	//randomization of Xmat rows
	a := rand.Perm(506) //outputs random indices for rows
	Xrand := mat.NewDense(len(records), 1+len(input_matrix[0])-1+len(onehot_neighborhood[0]), nil)
	Yrand := mat.NewDense(len(records), 1, nil)
	for t := 0; t < 506; t++ {
		Yrand.Set(t, 0, Ymat.At(a[t], 0))
		for u := 0; u < 107; u++ {
			Xrand.Set(t, u, Xmat.At(a[t], u))
		}
	}

	Xtrain := Xrand.Slice(0, 400, 0, 107)
	Xtest := Xrand.Slice(400, 506, 0, 107)
	Ytrain := Yrand.Slice(0, 400, 0, 1)
	Ytest := Yrand.Slice(400, 506, 0, 1)
	Yave := mat.Sum(Ytest) / 106
	Ymean := mat.NewDense(106, 1, nil)
	for i := 0; i < 106; i++ {
		Ymean.Set(i, 0, float64(Yave))
	}

	switch model {
	case 1:
		var Betaprime mat.Dense
		var Betaprim mat.Dense
		var Ypred mat.Dense
		var Ypred2 mat.Dense
		var xx mat.Dense
		var ixx mat.Dense
		var xy mat.Dense
		var e mat.Dense
		var e0 mat.Dense
		var e2 mat.Dense
		var e20 mat.Dense
		var yYmean mat.Dense
		var yYmean0 mat.Dense
		var yYmean2 mat.Dense
		var yYmean20 mat.Dense
		//Training with betaprime = ((XtX)^-1)*(XtY)
		xx.Mul(Xtrain.T(), Xtrain) // XtX
		ixx.Inverse(&xx)           //(XtX)^-1
		xy.Mul(Xtrain.T(), Ytrain) //(XtY)
		Betaprime.Mul(&ixx, &xy)
		//Testing
		Ypred.Mul(Xtest, &Betaprime)
		e.Sub(Ytest, &Ypred)
		rmse := e.Norm(2)
		//r2 calculation
		e2.Mul(e.T(), &e)
		yYmean.Sub(Ytest, Ymean)
		yYmean2.Mul(yYmean.T(), &yYmean)
		r2 := 1 - e2.At(0, 0)/yYmean2.At(0, 0)

		Betaprim.Solve(Xtrain, Ytrain)
		Ypred2.Mul(Xtest, &Betaprim)
		e0.Sub(Ytest, &Ypred)
		rmse2 := e0.Norm(2)
		//r2 calculation
		e20.Mul(e0.T(), &e0)
		yYmean0.Sub(Ytest, Ymean)
		yYmean20.Mul(yYmean.T(), &yYmean0)
		r22 := 1 - e2.At(0, 0)/yYmean20.At(0, 0)

		fmt.Println(rmse)
		fmt.Println(r2)
		fmt.Println(rmse2)
		fmt.Println(r22)
		fmt.Println(Ytest)
		//fmt.Println(e0)
	case 2:
		var Beta mat.Dense
		var xx mat.Dense
		var ixxi0 mat.Dense
		var xy mat.Dense
		var i0 mat.Dense
		var alphai0 mat.Dense
		var xxi0 mat.Dense
		xx.Mul(Xmat.T(), Xmat)
		i0.Pow(&xx, 0)
		alpha := mat.NewDense(107, 107, nil)
		for i := 0; i < 107; i++ {
			for j := 0; j < 107; j++ {
				alpha.Set(i, j, 0.51)
			}
		}
		fmt.Println(alpha.Dims())
		fmt.Println(i0.Dims())
		alphai0.Mul(&i0, alpha)
		fmt.Println(xx.Dims())
		fmt.Println(alphai0.Dims())
		xxi0.Add(&xx, &alphai0)
		ixxi0.Inverse(&xxi0)
		xy.Mul(Xmat.T(), Ymat)
		Beta.Mul(&ixxi0, &xy)
		fmt.Println(Beta)

	case 3:
		var Beta mat.Dense
		fmt.Println(Beta)
	case 4:
		var Beta mat.Dense
		fmt.Println(Beta)
	}

}
