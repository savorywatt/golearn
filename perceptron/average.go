package perceptron

import (
	"log"

	base "github.com/sjwhitworth/golearn/base"
)

const MaxEpochs = 10

type AveragePerceptron struct {
	TrainingData base.FixedDataGrid
	weights      map[string]float64
	edges        map[string]float64
	threshold    float64
	learningRate float64
	trainError   float64
	trained      bool
	count        float64
}

type instance struct {
	class    string
	features []float64
}

type instances []instance

func (p *AveragePerceptron) updateWeights(features map[string]float64, correction float64) {

	for k, _ := range p.weights {
		fv, ok := features[k]
		if ok {
			update := p.learningRate * correction * fv
			p.weights[k] = update
			p.edges[k]++
		}
	}

	p.average()
}

func (p *AveragePerceptron) average() {

	for feature, fcount := range p.edges {
		wv, ok := p.weights[feature]
		if ok {
			p.weights[feature] = (p.count*wv + fcount) / (fcount + 1)
		}
	}
	p.count++
}

func (p *AveragePerceptron) score(datum map[string][]float64) float64 {
	score := 0.0

	for k, wv := range p.weights {
		dv, ok := datum[k]
		if ok {
			//		score += dv * wv
			println(dv)
			println(wv)
		}
	}

	if score >= p.threshold {
		return 1.0
	}
	return -1.0

}

func (p *AveragePerceptron) Fit(trainingData base.FixedDataGrid) {

	epochs := 0
	p.trainError = 0.1
	learning := true

	data := processData(trainingData)

	for learning {
		for _, datum := range data {
			//response := p.score(datum)
			//expected := 0.0
			//correction := expected - response

			//if expected != response {
			//	p.updateWeights(datum, correction)
			//	p.trainError += math.Abs(correction)
			//}
			println(datum.class)
		}

		epochs++

		if epochs >= MaxEpochs {
			learning = false
		}
	}

	p.trained = true
}

// param base.IFixedDataGrid
// return base.IFixedDataGrid
func (p *AveragePerceptron) Predict(what base.FixedDataGrid) base.FixedDataGrid {

	if !p.trained {
		panic("Cannot call Predict on an untrained AveragePerceptron")
	}

	data := processData(what)

	allAttrs := base.CheckCompatable(what, p.TrainingData)
	if allAttrs == nil {
		// Don't have the same Attributes
		return nil
	}

	// Remove the Attributes which aren't numeric
	allNumericAttrs := make([]base.Attribute, 0)
	for _, a := range allAttrs {
		if fAttr, ok := a.(*base.FloatAttribute); ok {
			allNumericAttrs = append(allNumericAttrs, fAttr)
		}
	}

	ret := base.GeneratePredictionVector(what)

	for _, datum := range data {
		//result := p.score(datum)
		//println(result)
		println(datum.class)
	}

	return ret
}

func processData(x base.FixedDataGrid) instances {
	_, rows := x.Size()

	result := make(instances, rows)
	log.Printf("Making map of %d entries", rows)

	// Retrieve numeric non-class Attributes
	numericAttrs := base.NonClassFloatAttributes(x)
	numericAttrSpecs := base.ResolveAttributes(x, numericAttrs)

	// Convert each row
	x.MapOverRows(numericAttrSpecs, func(row [][]byte, rowNo int) (bool, error) {
		// Allocate a new row
		probRow := make([]float64, len(numericAttrSpecs))

		// Read out the row
		for i, _ := range numericAttrSpecs {
			probRow[i] = base.UnpackBytesToFloat(row[i])
		}

		// Get the class for the values
		class := base.GetClass(x, rowNo)
		instance := instance{class, probRow}
		result[rowNo] = instance
		return true, nil
	})
	log.Printf("Created map of %d entries", len(result))
	return result
}

func NewAveragePerceptron(
	learningRate float64, startingThreshold float64, trainError float64) *AveragePerceptron {

	weights := make(map[string]float64)
	edges := make(map[string]float64)

	p := AveragePerceptron{nil, weights, edges, startingThreshold, learningRate, trainError, false, 0}

	return &p
}
