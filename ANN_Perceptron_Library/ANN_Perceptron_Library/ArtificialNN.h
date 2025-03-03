#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ActivationFunction.h"
#include "NetworkLayer.h"

#define  ARTIFICIALNN_API __declspec(dllexport)

class ActivationFunction;

ARTIFICIALNN_API class ArtificialNN
{
public:

	// Number of input neurons in the network
	int mNumInputs;
	// Number of output neurons in the network
	int mNumOutputs;
	// Number of hidden layers in the network
	int mNumHidden;
	// Number of neurons per hidden layer
	int mNumPerHidden;
	// Learning rate for adjusting weights during training
	double mLearningRate;

	// Vector of layers in the network, using smart pointers for automatic memory management
	std::vector<std::shared_ptr<NetworkLayer>> mLayers;

	// Constructor to initialize the neural network with specific parameters
	ARTIFICIALNN_API ArtificialNN(int _numberInput, int _numberOutput,
				 int _numberHiddenLayer, int _NumberNeuronPerHiddenLayer, double _learningRate,
				 Activation_Function _af_HiddenLayer, Activation_Function _af_OutputLayer);

	// Destructor
	ARTIFICIALNN_API ~ArtificialNN();

	// Processes input through the network and optionally updates weights based on desired output
	ARTIFICIALNN_API std::vector<double> Go(std::vector<double> _inputValues, std::vector<double> _desiredOutput, bool _updateWeight = true);

	// Trains the network on a given set of inputs and desired outputs
	ARTIFICIALNN_API std::vector<double> Train(std::vector<double> _inputValues, std::vector<double> _desiredOutput);

	// Calculates the output of the network for a given set of inputs
	ARTIFICIALNN_API std::vector<double> CalcOutput(std::vector<double> _inputValues, std::vector<double> _desiredOutput);

	// Prints the weights and bias of all neurons in the network
	ARTIFICIALNN_API std::string PrintWeightsBias();

	// Saves the current weights and bias of the entire network to a file
	ARTIFICIALNN_API void SaveWeightsBias();

	// Loads weights and bias for the entire network from a file
	ARTIFICIALNN_API void LoadWeightsBias();

private:

	// Activation functions for hidden and output layers
	Activation_Function activationFunctionHiddenLayer;
	Activation_Function activationFunctionOutputLayer;

	// Updates the weights of the network based on the output and desired output
	ARTIFICIALNN_API void UpdateWeights(std::vector<double> _outputs, std::vector<double> _desiredOutput);

	// Activation functions used by neurons in the network
	ARTIFICIALNN_API double ActivationFunctionH(double _value);
	ARTIFICIALNN_API double ActivationFunctionO(double _value);

	// Specific activation function implementations
	ARTIFICIALNN_API double Step(double _value);		// Binary step function
	ARTIFICIALNN_API double TanH(double _value);		// Hyperbolic tangent function
	ARTIFICIALNN_API double Sigmoid(double _value);		// Sigmoid function
	ARTIFICIALNN_API double ReLU(double _value);		// Rectified Linear Unit function
	ARTIFICIALNN_API double LeakyReLU(double _value);	// Leaky Rectified Linear Unit function

	// Calculates the derivative of the activation function, used during backpropagation
	ARTIFICIALNN_API double Derivated_Activation_Function(Activation_Function _af, double _value);
};

