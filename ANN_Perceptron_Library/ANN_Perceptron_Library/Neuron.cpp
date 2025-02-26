#include "pch.h"
#include "Neuron.h"

#include <random>

Neuron::Neuron(int _nInputs)
{
    mNumInputs = _nInputs;
    InitializeWeightsAndBias();
}

NEURON_API void Neuron::InitializeWeightsAndBias()
{
    weights.resize(mNumInputs);
    weights = { RandomDoubleNumber(-1, 1), RandomDoubleNumber(-1, 1)};
    mBias = RandomDoubleNumber(-1, 1);
}

NEURON_API double Neuron::RandomDoubleNumber(double _lowerLimit, double _upperLimit)
{
   // Seed for random generation
    std::random_device rd;

    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(_lowerLimit, _upperLimit);

    return dis(gen);
}


