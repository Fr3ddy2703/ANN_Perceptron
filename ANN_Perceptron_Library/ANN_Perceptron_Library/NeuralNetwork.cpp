#include "pch.h"
#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(std::vector<std::pair<int, ActivationFType>> _layerCreation, const double& _learningRate)
	: mLearningRate(_learningRate)
{
	for (int i = 1; i < _layerCreation.size(); i++)
			mNetworkLayers.emplace_back(NetworkLayer(_layerCreation[i].first, _layerCreation[i].second, _layerCreation[i - 1].first));
}

std::vector<double> NeuralNetwork::CalculateNetworkOutput(const std::vector<double>& _inputs)
{
	std::vector<double> output = _inputs;

	for (NetworkLayer& layer : mNetworkLayers)
	{
		output = layer.CalculateLayerOutput(output);
	}
	return output;
}

void NeuralNetwork::Train(const std::vector<std::vector<double>>& _trainingInputs,
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

			totalLoss += loss;

			Backropagate(_trainingOutputs[i]);
			UpdateNetworkWeights(_trainingInputs[i]);
		}

		if (_print)
				std::cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / _trainingInputs.size() << std::endl;
		
	}
}

ANN_API int NeuralNetwork::SelectionActionSoftmax(const std::vector<double>& _state)
{
	std::vector<double> QValues = CalculateNetworkOutput(_state);

	std::vector<double> probabilites(QValues.size());

	double sumExp = 0.0;

	for (double q : QValues)
		sumExp += std::exp(q / mTau);
	
	for (size_t i = 0; i < QValues.size(); i++)
		probabilites[i] = std::exp(QValues[i] / mTau / sumExp);

	double r = (double)rand() / RAND_MAX;
	double cumulativeProbability = 0.0;

	for (size_t i = 0; i < probabilites.size(); i++)
	{
		cumulativeProbability += probabilites[i];
		if (r <= cumulativeProbability)
			return static_cast<int>(i);
	}
	return static_cast<int>(QValues.size() - 1);
}

void NeuralNetwork::StoreExperience(const std::vector<double>& _state, int _action, double _reward, 
									const std::vector<double>& _nextState, 
									bool _terminal)
{
	if (mReplayBuffer.size() >= mCapacity)
		mReplayBuffer.erase(mReplayBuffer.begin());

	mReplayBuffer.push_back({_state, _action, 
		_reward, _nextState, _terminal });

}

void NeuralNetwork::TrainQLearning()
{
	if (mReplayBuffer.size() < 32)
		return;

	std::vector<Experience> batch;

	for (int i = 0; i < 32; i++)
		batch.push_back(mReplayBuffer[rand() % mReplayBuffer.size()]);

	for (Experience& exp : batch)
	{
		std::vector<double> QValue = CalculateNetworkOutput(exp.mState);
		double targetQ = exp.mReward;

		if (!exp.mTerminal)
		{
			std::vector<double> nextQValues = CalculateNetworkOutput(exp.mNextState);
			targetQ += mDiscountFactor * (*std::max_element(nextQValues.begin(), nextQValues.end()));
		}

		QValue[exp.mAction] = targetQ;

		Backropagate(QValue);
		UpdateNetworkWeights(exp.mState);
	}

	if (mTau > mTauMin)
		mTau *= mTauDecay;
}

void NeuralNetwork::Backropagate(const std::vector<double>& _expectedOutputs)
{
	mNetworkLayers.back().ComputeErrorGradientLayer(_expectedOutputs);

	for (int i = static_cast<int>(mNetworkLayers.size() - 2); i >= 0; i--)
		mNetworkLayers[i].ComputeErrorGradientLayer(mNetworkLayers[i + 1]);
}

void NeuralNetwork::UpdateNetworkWeights(const std::vector<double>& _trainingInputs)
{
	std::vector<double> previousOutputs = _trainingInputs;

	for (NetworkLayer& layer : mNetworkLayers)
	{
		layer.UpdateLayerWeights(mLearningRate, previousOutputs);
		previousOutputs = layer.GetOutput();
	}
}
