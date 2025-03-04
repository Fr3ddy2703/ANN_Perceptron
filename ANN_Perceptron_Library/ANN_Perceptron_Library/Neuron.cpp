#include "pch.h"
#include "Neuron.h"

#include <random>

Neuron::Neuron(const double& _initBias, std::vector<double>& _initWeight)
	: mBias(_initBias), mWeights(_initWeight), mOutput(0), mErrorGradient(0),
		mNetInput(_initWeight.size())
{
}

ANN_API double Neuron::CalculateOutput(const std::vector<double>& _inputs, ActivationFType _afType)
{
	if (_inputs.size() != mWeights.size())
	{
		throw std::invalid_argument("Input size must match weight size!");
	}

	double output = mBias;
	for (size_t i = 0; i < _inputs.size(); i++)
	{
		output += _inputs[i] * mWeights[i];
	}

	mNetInput += output;

	mOutput = Activation_Function::ExecuteActivationFunction(output, _afType);

	return mOutput;
}

ANN_API void Neuron::ComputeErrorGradient(const double& _errorSignal, ActivationFType _afType)
{
	double activationDerivative = Activation_Function::ExecuteActivationFunctionDerivative(mOutput, _afType);

	mErrorGradient = _errorSignal * activationDerivative;
}

ANN_API void Neuron::UpdateWeights(const double& _learningRate, const std::vector<double>& _prevOutput)
{
	for (size_t i = 0; i < mWeights.size(); i++)
	{
		mWeights[i] -= _learningRate * mErrorGradient * _prevOutput[i];
	}

	mBias -= _learningRate * mErrorGradient;
}

ANN_API double Neuron::GetErrorGradient()
{
	return mErrorGradient;
}

ANN_API double Neuron::GetWeight(const int& _index)
{
	if (_index >= 0 && _index < mWeights.size())
	{
		return mWeights[_index];
	}
	return 0.0;
}

ANN_API double Neuron::GetOutput()
{
	return mOutput;
}
