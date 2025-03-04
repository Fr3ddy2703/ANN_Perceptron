#include "pch.h"
#include "NetworkLayer.h"

#include <random>

NetworkLayer::NetworkLayer(const int& _numNeurons, ActivationFType _aft, const int& _numExpectedInputs)
	: mLayerActivationFunctionType(_aft)
{
	for (int i = 0; i < _numNeurons; ++i)
	{
		std::vector<double> RandomWeights(_numExpectedInputs);

		for (int j = 0; j < _numExpectedInputs; j++)
				RandomWeights[j] = GetRandomDouble(-0.5, 0.5);

		double RandomBias = GetRandomDouble(-0.5, 0.5);
		mLayerNeurons.push_back(std::make_shared<Neuron>(RandomBias, RandomWeights));
	}
}

ANN_API std::vector<double> NetworkLayer::CalculateLayerOutput(const std::vector<double>& _inputs)
{
	if (mLayerOutputs.size() != mLayerNeurons.size())
			mLayerOutputs.resize(mLayerNeurons.size());

	for (size_t i = 0; i < mLayerNeurons.size(); i++)
			mLayerOutputs[i] = mLayerNeurons[i]->CalculateOutput(_inputs, mLayerActivationFunctionType);

	return mLayerOutputs;
}

ANN_API void NetworkLayer::ComputeErrorGradientLayer(const NetworkLayer& _nextLayer)
{
	for (int i = 0; i < static_cast<int>(mLayerNeurons.size()); i++)
	{
		double GradientSum = 0.0;

		for (const std::shared_ptr<Neuron>& nextNeuron : _nextLayer.GetNeurons())
		{
			GradientSum += nextNeuron->GetWeight(i) * nextNeuron->GetErrorGradient();
		}
		mLayerNeurons[i]->ComputeErrorGradient(GradientSum, mLayerActivationFunctionType);
	}
}

ANN_API void NetworkLayer::ComputeErrorGradientLayer(const std::vector<double>& _expectedOutput)
{
	for(size_t i = 0; i < mLayerNeurons.size(); i++)
			mLayerNeurons[i]->ComputeErrorGradient(_expectedOutput[i] - mLayerNeurons[i]->GetOutput(), mLayerActivationFunctionType);
}

ANN_API void NetworkLayer::UpdateLayerWeights(const double& _learningRate, const std::vector<double>& _previousOutput)
{
	for (const std::shared_ptr<Neuron>& neuron : mLayerNeurons)
			neuron->UpdateWeights(_learningRate, _previousOutput);
}

ANN_API std::vector<double> NetworkLayer::GetOutput()
{
	std::vector<double> outputs;

	for(const std::shared_ptr<Neuron>& neuron : mLayerNeurons)
			outputs.push_back(neuron->GetOutput());

	return outputs;
}

ANN_API const std::vector<std::shared_ptr<Neuron>>& NetworkLayer::GetNeurons() const
{
	return mLayerNeurons;
}

ANN_API std::vector<double> NetworkLayer::GetLayerErrorGradient()
{
	std::vector<double> gradients;

	for (const std::shared_ptr<Neuron>& neuron : mLayerNeurons)
			gradients.push_back(neuron->GetErrorGradient());

	return gradients;
}

ANN_API double NetworkLayer::GetRandomDouble(const double& _lowerBound, const double& _upperBound)
{
	std::random_device rd;

	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(_lowerBound, _upperBound);

	return dis(gen);
}
