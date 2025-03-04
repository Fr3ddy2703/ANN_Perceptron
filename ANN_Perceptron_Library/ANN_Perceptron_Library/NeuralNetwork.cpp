#include "pch.h"
#include "NeuralNetwork.h"

#include <iostream>

NeuralNetwork::NeuralNetwork(std::vector<std::pair<int, ActivationFType>> _layerCreation, const double& _learningRate)
	: mLearningRate(_learningRate)
{
	for (int i = 1; i < _layerCreation.size(); i++)
			mNetworkLayers.emplace_back(NetworkLayer(_layerCreation[i].first, _layerCreation[i].second, _layerCreation[i - 1].first));
}

ANN_API std::vector<double> NeuralNetwork::CalculateNetworkOutput(const std::vector<double>& _inputs)
{
	std::vector<double> output = _inputs;

	for (NetworkLayer& layer : mNetworkLayers)
	{
		output = layer.CalculateLayerOutput(output);
	}
	return output;
}

ANN_API void NeuralNetwork::Train(const std::vector<std::vector<double>>& _trainingInputs,
									const std::vector<std::vector<double>>& _trainingOutputs, 
									const int& _epochs, bool _print)
{
	for (int epoch = 0; epoch < _epochs; epoch++)
	{
		double totalLoss = 0.0;

		for (size_t i = 0; i < _trainingInputs.size(); i++)
		{
			std::vector<double> output = CalculateNetworkOutput(_trainingInputs[i]);

			double loss = 0.0;

			for (size_t j = 0; j < output.size(); j++)
			{
				double diff = _trainingOutputs[i][j] - output[j];
				loss += diff * diff;
			}

			totalLoss += loss / output.size();

			Backropagate(_trainingOutputs[i]);
			UpdateNetworkWeights(_trainingInputs[i]);
		}

		if (_print)
				std::cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / _trainingInputs.size() << std::endl;
		
	}
}

ANN_API void NeuralNetwork::Backropagate(const std::vector<double>& _expectedOutputs)
{
	mNetworkLayers.back().ComputeErrorGradientLayer(_expectedOutputs);

	for (int i = static_cast<int>(mNetworkLayers.size() - 2); i >= 0; i--)
		mNetworkLayers[i].ComputeErrorGradientLayer(mNetworkLayers[i + 1]);
}

ANN_API void NeuralNetwork::UpdateNetworkWeights(const std::vector<double>& _trainingInputs)
{
	std::vector<double> previousOutputs = _trainingInputs;

	for (NetworkLayer& layer : mNetworkLayers)
	{
		layer.UpdateLayerWeights(mLearningRate, previousOutputs);
		previousOutputs = layer.GetOutput();
	}
}
