#include "pch.h"
#include "ANN_Perceptron.h"
#include <iostream>
#include <random>

Perceptron::Perceptron(std::vector<TrainingSet_Perceptron>& _ts) : ts(_ts)
{
	InitializeWeightsAndBias();
}

ANN_PERCEPTRON_API void Perceptron::Train()
{
	totalError = 0;
	for (TrainingSet_Perceptron trainingSet : ts)
	{
		double calculatedOutput = (trainingSet.input[0] * weights[0]) + (trainingSet.input[1] * weights[1]) + bias;
		calculatedOutput = (calculatedOutput >= 0) ? 1 : 0;
		double error = trainingSet.desiredOutput - calculatedOutput;

		std::cout << "Input: " << trainingSet.input[0] << " " << trainingSet.input[1]
                  << " | Output: " << calculatedOutput
                  << " | Expected: " << trainingSet.desiredOutput << std::endl;

		weights[0] = weights[0] + error * trainingSet.input[0];
		weights[1] = weights[1] + error * trainingSet.input[1];
		bias = bias + error;
	}
}

ANN_PERCEPTRON_API void Perceptron::Train(int _epochs)
{
	for (int i = 0; i < _epochs; i++)
	{
		Train();
	}
}

ANN_PERCEPTRON_API void Perceptron::InitializeWeightsAndBias()
{
	weights = { RandomDoubleNumber(-1, 1), RandomDoubleNumber(-1, 1) };
	bias = RandomDoubleNumber(-1, 1);
}

ANN_PERCEPTRON_API double Perceptron::CalcOutput(double _input1, double _input2)
{
	return (_input1 * weights[0]) + (_input2 * weights[1]) + bias;
}

ANN_PERCEPTRON_API double Perceptron::RandomDoubleNumber(double _lowerLimit, double _upperLimit)
{
	// Seed for random generation
	std::random_device rd;

	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(_lowerLimit, _upperLimit);

	return dis(gen);
}




