#pragma once

#include "Defines.h"
#include <complex>
#include <memory>
#include <utility>

enum ActivationFType
{
	AF_Identity,
	AF_BinaryStep,
	AF_SiLU,
	AF_Hyperbolic,
	AF_RectifiedLinear,
	AF_LeakyRectified
};

class Activation_Function
{
public:

	ANN_API static double ExecuteActivationFunction(double _calculatedOutput, ActivationFType _aft)
	{
		switch (_aft)
		{
		case ActivationFType::AF_Identity:
			return Identity(_calculatedOutput);
		case ActivationFType::AF_BinaryStep:
			return BinaryStep(_calculatedOutput);
		case ActivationFType::AF_SiLU:
			return SiLU(_calculatedOutput);
		case ActivationFType::AF_Hyperbolic:
			return Hyperbolic(_calculatedOutput);
		case ActivationFType::AF_RectifiedLinear:
			return RectifiedLinear(_calculatedOutput);
		case ActivationFType::AF_LeakyRectified:
			return LeakyRectified(_calculatedOutput);

		}

		throw std::invalid_argument("No Activation function Match");
		return -1;
	}

	ANN_API static double ExecuteActivationFunctionDerivative(double _calculatedOutput, ActivationFType _aft)
	{
		switch (_aft)
		{
		case ActivationFType::AF_Identity:
			return IdentityDeriv();
		case ActivationFType::AF_BinaryStep:
			return BinaryStepDeriv();
		case ActivationFType::AF_SiLU:
			return SiLUDeriv(_calculatedOutput);
		case ActivationFType::AF_Hyperbolic:
			return HyperbolicDeriv(_calculatedOutput);
		case ActivationFType::AF_RectifiedLinear:
			return RectifiedLinearDeriv(_calculatedOutput);
		case ActivationFType::AF_LeakyRectified:
			return LeakyRectifiedDeriv(_calculatedOutput);
		}

		throw std::invalid_argument("No Derivative Activation function Match");
		return -1;
	}

private:

	ANN_API static double Identity(double _calculatedOutput)
	{
		return _calculatedOutput;
	}
	ANN_API static double IdentityDeriv()
	{
		return 1;
	}
	ANN_API static double BinaryStep(double _calculatedOutput)
	{
		return _calculatedOutput >= 0 ? 1 : 0;
	}
	ANN_API static double BinaryStepDeriv()
	{
		return 0;
	}
	ANN_API static double SiLU(double _calculatedOutput)
	{
		return _calculatedOutput / (1.0 + std::exp(-_calculatedOutput));
	}
	ANN_API static double SiLUDeriv(double _calculatedOutput)
	{
		double top = 1.0 + std::exp(-_calculatedOutput) + _calculatedOutput * std::exp(-_calculatedOutput);
		double bottom = std::pow(1.0 + std::exp(-_calculatedOutput),2);
		return top / bottom;
	}
	ANN_API static double Hyperbolic(double _calculatedOutput)
	{
		return tanh(_calculatedOutput);
	}
	ANN_API static double HyperbolicDeriv(double _calculatedOutput)
	{
		double hyper = Hyperbolic(_calculatedOutput);
		return 1 - std::pow(hyper, 2);
	}
	ANN_API static double RectifiedLinear(double _calculatedOutput)
	{
		return _calculatedOutput > 0 ? _calculatedOutput : 0;
	}
	ANN_API static double RectifiedLinearDeriv(double _calculatedOutput)
	{
		return _calculatedOutput > 0 ? 1 : 0;
	}
	ANN_API static double LeakyRectified(double _calculatedOutput)
	{
		return _calculatedOutput <= 0 ? 0.01 * _calculatedOutput : _calculatedOutput;
	}
	ANN_API static double LeakyRectifiedDeriv(double _calculatedOutput)
	{
		return _calculatedOutput < 0 ? 0.01 : 1;
	}
};