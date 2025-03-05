#include "pch.h"
#include "Neuron.h"

Neuron::Neuron(const double& _initBias, std::vector<double>& _initWeight)
	: mBias(_initBias), mWeights(std::move(_initWeight))
{
}

double Neuron::CalculateOutput(const std::vector<double>& _inputs, ActivationFType _afType)
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

	mNetInput = output;

	mOutput = Activation_Function::ExecuteActivationFunction(output, _afType);

	return mOutput;
}

void Neuron::ComputeErrorGradient(const double& _errorSignal, ActivationFType _afType)
{
	double activationDerivative = Activation_Function::ExecuteActivationFunctionDerivative(mNetInput, _afType);

	mErrorGradient = _errorSignal * activationDerivative;
}

void Neuron::UpdateWeights(const double& _learningRate, const std::vector<double>& _prevOutput)
{
	for (size_t i = 0; i < mWeights.size(); i++)
	{
		mWeights[i] += _learningRate * mErrorGradient * _prevOutput[i];
	}

	mBias += _learningRate * mErrorGradient;
}

double Neuron::GetErrorGradient()
{
	return mErrorGradient;
}

double Neuron::GetWeight(const int& _index)
{
	if (_index >= static_cast<int>(mWeights.size() || _index < 0))
	{
		return mWeights[_index];
	}
	return 0.0;
}

double Neuron::GetOutput()
{
	return mOutput;
}
