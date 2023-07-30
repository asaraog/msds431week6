package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

var model = flag.Int("model", 1, "model number")

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
	flag.Parse()
	//Reads in each line of input as an array of strings
	file, _ := os.Open("boston.csv")

	reader := csv.NewReader(file)
	record, _ := reader.ReadAll()

	//headers := records[1][:]
	records := record[1:][:]
	matrix := make([][]float64, len(records))
	for i := range records {
		matrix[i] = make([]float64, len(records[0]))
		for y := range records[i] {
			matrix[i][y], _ = strconv.ParseFloat(records[i][y], 64)
		}
	}

	//Makes Y matrix
	Y := make([]float64, len(records))
	for m := range Y {
		Y[m] = matrix[m][len(matrix[0])-1]
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

	// X := make([][]float64, len(records))
	// for w := range X {
	// 	X[w] = make([]float64, 1+len(matrix[0])-1+len(onehot_neighborhood[0]))
	// }

	// for n := range X {
	// 	//sets Beta0 or constant term
	// 	X[n][1] = 1

	// 	//sets neighborhood and other matrix values
	// 	for o := range onehot_neighborhood[0] {
	// 		X[n][o+1] = onehot_neighborhood[n][o]
	// 	}
	// 	for p := range matrix[0][:len(matrix[0])-1] {
	// 		X[n][p+len(onehot_neighborhood[0])+1] = matrix[n][p]
	// 	}
	// }

	Xmat := mat.NewDense(len(records), 1+len(matrix[0])-1+len(onehot_neighborhood[0]), nil)
	for n := range records {
		//sets Beta0 or constant term
		Xmat.Set(n, 1, 1)

		//sets neighborhood and other matrix values
		for o := range onehot_neighborhood[0] {
			Xmat.Set(n, o+1, onehot_neighborhood[n][o])
		}
		for p := range matrix[0][:len(matrix[0])-1] {
			Xmat.Set(n, p+len(onehot_neighborhood[0])+1, matrix[n][p])
		}
	}
	fmt.Println(Xmat)
}
