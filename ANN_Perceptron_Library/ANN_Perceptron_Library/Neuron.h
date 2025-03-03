#pragma once

#include "ActivationFunction.h"
#include "Defines.h"
#include <vector>


enum ActivationFTYpe;

// Define a class representing a single neuron in a neural network
ANN_API class Neuron
{
public:

private:
	double mNetInput = 0;			// Number of inputs the neuron receives
	double mBias;			// Bias value for the neuron, used in its activation function
	double mOutput = 0;			// The output value of the neuron after processing inputs
	double mErrorGradient = 0;		// The gradient of the error for this neuron, used during backpropagation

	std::vector<double> mWeights; // Dynamic array of weights for each input


public:

	// Constructor that initializes a neuron with a specific number of inputs
	ANN_API Neuron(const double& _initBias, std::vector<double>& _initWeight);

	// Destructor
	ANN_API ~Neuron() = default;

	// Calculated the output
	ANN_API double CalculateOutput(const std::vector<double>& _inputs, ActivationFType _afType);

	// Computes the error gradient
	ANN_API void ComputeErrorGradient(const double& _errorSignal, ActivationFType _afType);

	// Updates the weights 
	ANN_API void UpdateWeights(const double& _learningRate, const std::vector<double>& _prevOutput);

	// Gets the error gradient
	ANN_API double GetErrorGradient();

	// Gets the weight
	ANN_API double GetWeight(const int& _index);

	// Gets the output
	ANN_API double GetOutput();

};


