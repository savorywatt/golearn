package perceptron

import (
	"path/filepath"
	"testing"

	"github.com/sjwhitworth/golearn/base"
)

func TestProcessData(t *testing.T) {
	absPath, _ := filepath.Abs("../examples/datasets/house-votes-84.csv")
	rawData, err := base.ParseCSVToInstances(absPath, true)
	trainData, _ := base.InstancesTrainTestSplit(rawData, 0.5)

	if err != nil {
		t.Fatal("Could not test processData. Could not load CSV")
	}

	if rawData == nil {
		t.Fatal("Could not test processData. Could not load CSV")
	}

	result := processData(trainData)
	_, size := trainData.Size()

	if len(result) != size {
		t.Errorf("Expected %d, Got %d", size, len(result))
	}
}

func TestCreateAveragePerceptron(t *testing.T) {

	a := NewAveragePerceptron(10, 1.2, 0.5, 0.3)

	if a == nil {

		t.Errorf("Unable to create average perceptron")
	}
}
