#include "../include/Neuron.h"
#include <random>

double generateRandomNumber(double min, double max)
{
    double random = static_cast<double>(rand()) / RAND_MAX;
    return min + random * (max - min);
}

Neuron::Neuron(int previousLayerSize)
{
    InitializeWeights(previousLayerSize);
}

void Neuron::InitializeWeights(int previousLayerSize)
{
    std::normal_distribution<double> distribution(0.0, 1.0);
    for(int i = 0; i < previousLayerSize + 1; i++)
    {
        Weights.push_back(generateRandomNumber(-1.0, 1.0));
    }
}
