package perceptron

import base "github.com/sjwhitworth/golearn/base"

const MaxEpochs = 10

type AveragePerceptron struct {
	TrainingData base.FixedDataGrid
	weights      []float64
	edges        []float64
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

func (p *AveragePerceptron) updateWeights(features []float64, correction float64) {

	for i, _ := range p.weights {
		fv := &features[i]
		if fv != nil {
			update := p.learningRate * correction * *fv
			p.weights[i] = update
			p.edges[i]++
		}
	}

	p.average()
}

func (p *AveragePerceptron) average() {

	for i, fcount := range p.edges {
		wv := &p.weights[i]
		if wv != nil {
			p.weights[i] = (p.count**wv + fcount) / (fcount + 1)
		}
	}
	p.count++
}

func (p *AveragePerceptron) score(datum instance) float64 {
	score := 0.0

	for i, wv := range p.weights {
		dv := &datum.features[i]
		if dv != nil {
			//		score += dv * wv
			println(*dv)
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
	return result
}

func NewAveragePerceptron(features int, learningRate float64, startingThreshold float64, trainError float64) *AveragePerceptron {

	weights := make([]float64, features)
	edges := make([]float64, features)

	p := AveragePerceptron{nil, weights, edges, startingThreshold, learningRate, trainError, false, 0}

	return &p
}
