#pragma once

#include <vector>

#define  NEURON_API __declspec(dllexport)
//#ifdef ANN_PERCEPTRON_EXPORTS
//#else
//#define ANN_PERCEPTRON_API __declspec(dllimport)
//#endif

// Define a class representing a single neuron in a neural network
NEURON_API class Neuron
{
public:
	int mNumInputs;			// Number of inputs the neuron receives
	double mBias;			// Bias value for the neuron, used in its activation function
	int mOutput;			// The output value of the neuron after processing inputs
	int mErrorGradient;		// The gradient of the error for this neuron, used during backpropagation
	int mN;					// The value before activation function is applied (also known as the net input)

	std::vector<double> weights; // Dynamic array of weights for each input
	std::vector<double> inputs; // Dynamic array of inputs received by the neuron

	// Saves the weights and bias of the current neuron to file
	NEURON_API void SaveWeightsBias();

	// Initializes the weights and bias of the perceptron to random values
	NEURON_API void InitializeWeightsAndBias();

	// Handling the random generations of numbers
	NEURON_API double RandomDoubleNumber(double _lowerLimit, double _upperLimit);

	// Loads the weights and bias of the current neuron from a file
	NEURON_API void LoadWeightsBias();

	// Constructor that initializes a neuron with a specific number of inputs
	NEURON_API Neuron(int _nInputs);

	// Destructor
	NEURON_API ~Neuron() = default;
};


