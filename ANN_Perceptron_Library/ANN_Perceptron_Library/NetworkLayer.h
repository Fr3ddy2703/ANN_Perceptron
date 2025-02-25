#pragma once

#include <memory>
#include <vector>

#include "Neuron.h"

#define  NETWORKLAYER_API __declspec(dllexport)

NETWORKLAYER_API class NetworkLayer
{
public:
	int numNeurons; // Holds the number of neurons in this layer

	// Vector of shared_ptr to Neuron objects
	std::vector<std::shared_ptr<Neuron>> mNeurons;

	// Constructor that initializes a network layer with a specified number of neurons,
	// each neuron having a specified number of inputs
	NETWORKLAYER_API NetworkLayer(int _nNeurons, int _numNeuronInputs);

	// Destructor
	NETWORKLAYER_API ~NetworkLayer();

	// Saves the weights and bias of the current layer to a file
	NETWORKLAYER_API void SaveWeightBias();

	// Loads the weights and bias of the current layer from a file
	NETWORKLAYER_API void LoadWeightsBias();
};

