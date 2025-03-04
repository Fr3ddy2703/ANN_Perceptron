#pragma once

#include "NetworkLayer.h"

class ActivationFunction;

ANN_API class NeuralNetwork
{

public:


private:

	// Learning rate for the ANN
	double mLearningRate;

	// Stores the numbers of network layers
	std::vector<NetworkLayer> mNetworkLayers;

public:

	ANN_API NeuralNetwork() = default;

	ANN_API NeuralNetwork(std::vector<std::pair<int, ActivationFType>> _layerCreation, const double& _learningRate = 0.06); 

	ANN_API ~NeuralNetwork() = default;

	ANN_API std::vector<double> CalculateNetworkOutput(const std::vector<double>& _inputs);
	
	ANN_API void Train(const std::vector<std::vector<double>>& _trainingInputs,
					   const std::vector<std::vector<double>>& _trainingOutputs,
					   const int& _epochs, bool _print);


private:

	ANN_API void Backropagate(const std::vector<double>& _expectedOutputs);

	ANN_API void UpdateNetworkWeights(const std::vector<double>& _trainingInputs);
};

