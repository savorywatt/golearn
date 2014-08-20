package perceptron

import "math"

const MaxEpochs = 10

type AveragePerceptron struct {
	weights      map[string]float64
	edges        map[string]float64
	threshold    float64
	learningRate float64
	trainError   float64
	trained      bool
	count        float64
}

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

func (p *AveragePerceptron) score(datum map[string]float64) float64 {
	score := 0.0

	for k, wv := range p.weights {
		dv, ok := datum[k]

		if ok {
			score += dv * wv
		}

	}

	if score >= p.threshold {
		return 1.0
	}

	return -1.0

}

func (p *AveragePerceptron) Fit() {

	epochs := 0
	p.trainError = 0.1
	learning := true

	data := processData()

	for learning {
		for _, datum := range data {

			response := p.score(datum)
			expected := 0.0

			correction := expected - response

			if expected != response {
				p.updateWeights(datum, correction)
				p.trainError += math.Abs(correction)
			}
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
func (p *AveragePerceptron) Predict() {

	if !p.trained {
		panic("Cannot call Predict on an untrained AveragePerceptron")
	}

	data := processData()

	for _, datum := range data {

		result := p.score(datum)
		println(result)
	}

}

func processData() []map[string]float64 {

	return make([]map[string]float64, 0)
}

func NewAveragePerceptron(
	learningRate float64, startingThreshold float64, trainError float64) *AveragePerceptron {

	weights := make(map[string]float64)
	edges := make(map[string]float64)

	p := AveragePerceptron{weights, edges, startingThreshold, learningRate, trainError, false, 0}

	return &p
}
