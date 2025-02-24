#pragma once
#include <string>
#include <vector>

#define  ANN_PERCEPTRON_API __declspec(dllexport)
//#ifdef ANN_PERCEPTRON_EXPORTS
//#else
//#define ANN_PERCEPTRON_API __declspec(dllimport)
//#endif

ANN_PERCEPTRON_API class TrainingSet_Perceptron
{
public:
	// Vector to store input values
	std::vector<double> input;

	// Expected output value for the given inputs
	double desiredOutput;
};

ANN_PERCEPTRON_API class Perceptron
{
private:
	// Tracks the total error during training
	double totalError = 0;

	// Calculates the dot product of weights and inputs, including bias
	ANN_PERCEPTRON_API double DotProductBias(std::vector<double> _weights, std::vector<double> _inputs);

	// Calculates the output of the perceptron for a given training set index
	ANN_PERCEPTRON_API double CalcOutput(int _input1);

	// Updates the weights of the perceptron based on the error
	ANN_PERCEPTRON_API void UpdateWeights(int _j);

public:

	// Initialize weights of the perceptron
	std::vector<double> weights = { 0.0, 0.0 };

	// Bias value of the perceptron
	double bias = 0;

	// Vector of training sets
	std::vector<TrainingSet_Perceptron> ts;

	// Default constructor
	ANN_PERCEPTRON_API Perceptron();

	// Constructor that takes a vector og training sets
	ANN_PERCEPTRON_API Perceptron(std::vector<TrainingSet_Perceptron>& _ts);

	// Destructor
	ANN_PERCEPTRON_API ~Perceptron() = default;

	// Trains the perceptron and returns a string indicating the status
	ANN_PERCEPTRON_API void Train();

	// Trains the perceptron for a specified number of epochs and returns a string indicating the status
	ANN_PERCEPTRON_API void Train(int _epochs);

	// Initializes the weights and bias of the perceptron to random values
	ANN_PERCEPTRON_API void InitializeWeightsAndBias();

	// Calculates the output of the perceptron for given input values
	ANN_PERCEPTRON_API double CalcOutput(double _input1, double _input2);

	// Handling the random generations of numbers
	ANN_PERCEPTRON_API double RandomDoubleNumber(double _lowerLimit, double _upperLimit);
};


