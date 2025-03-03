#pragma once

#include "Neuron.h"

class NetworkLayer
{
public:

private:

	std::vector<std::shared_ptr<Neuron>> mLayerNeurons;
	std::vector<double> mLayerOutputs;
	ActivationFType mLayerActivationFunctionType;

public:

	ANN_API NetworkLayer(const int& _numNeurons, ActivationFType _aft, const int& _numExpectedInputs);

	ANN_API ~NetworkLayer() = default;

	ANN_API std::vector<double> CalculateLayerOutput(const std::vector<double>& _inputs);

	ANN_API void ComputeErrorGradientLayer(const NetworkLayer& _nextLayer);

	ANN_API void ComputeErrorGradientLayer(const std::vector<double>& _expectedOutput);

	ANN_API void UpdateLayerWeights(const double& _learningRate, const std::vector<double>& _previousOutput);

	ANN_API std::vector<double> GetOutPut();

private:

	ANN_API const std::vector<std::shared_ptr<Neuron>>& GetNeurons() const;

	ANN_API std::vector<double> GetLayerErrorGradient();

	ANN_API double GetRandomDouble(const double& _lowerBound, const double& _upperBound);
};

