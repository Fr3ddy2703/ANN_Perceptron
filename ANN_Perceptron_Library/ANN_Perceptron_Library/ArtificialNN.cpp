#include "pch.h"
#include "ArtificialNN.h"

ArtificialNN::ArtificialNN(int _numberInput, int _numberOutput, int _numberHiddenLayer, int _NumberNeuronPerHiddenLayer, double _learningRate, Activation_Function _af_HiddenLayer, Activation_Function _af_OutputLayer)
{
	mNumInputs = _numberInput;
	mNumOutputs = _numberOutput;
	mNumHidden = _numberHiddenLayer;
	mNumPerHidden = _NumberNeuronPerHiddenLayer;
	mLearningRate = _learningRate;

	activationFunctionHiddenLayer = _af_HiddenLayer;
	activationFunctionOutputLayer = _af_OutputLayer;
	
}

ARTIFICIALNN_API std::vector<double> ArtificialNN::Go(std::vector<double> _inputValues, std::vector<double> _desiredOutput, bool _updateWeight)
{
	std::vector<double> output = _inputValues;

	for (auto& layer : mLayers)
	{
		output = layer->
	}
}

ARTIFICIALNN_API std::vector<double> ArtificialNN::Train(std::vector<double> _inputValues, std::vector<double> _desiredOutput)
{

}

ARTIFICIALNN_API std::vector<double> ArtificialNN::CalcOutput(std::vector<double> _inputValues, std::vector<double> _desiredOutput)
{

}
