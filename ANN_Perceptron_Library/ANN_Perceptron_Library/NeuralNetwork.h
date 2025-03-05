#pragma once

#include "NetworkLayer.h"

class ActivationFunction;

struct Experience
{
	std::vector<double> mState;
	int mAction;
	double mReward;
	std::vector<double> mNextState;
	bool mTerminal;
};

ANN_API class NeuralNetwork
{

public:


private:

	// Learning rate for the ANN
	double mLearningRate;
	double mDiscountFactor = 0.99;
	double mTau = 1.0;
	double mTauDecay = 0.995;
	double mTauMin = 0.1;

	size_t mCapacity = 10000;

	std::vector<Experience> mReplayBuffer;

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

	ANN_API int SelectionActionSoftmax(const std::vector<double>& _state);

	ANN_API void StoreExperience(const std::vector<double>& _state, int _action, 
								double _reward, const std::vector<double>& _nextState,
								bool _terminal);

	ANN_API void TrainQLearning();


private:

	ANN_API void Backropagate(const std::vector<double>& _expectedOutputs);

	ANN_API void UpdateNetworkWeights(const std::vector<double>& _trainingInputs);
};

