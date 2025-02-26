#include "pch.h"
#include "NetworkLayer.h"

NetworkLayer::NetworkLayer(int _nNeurons, int _numNeuronInputs)
{
	numNeurons = _nNeurons;
	mNeurons.reserve(numNeurons);

	mNeurons.push_back(std::make_shared<Neuron>(_numNeuronInputs));
}
