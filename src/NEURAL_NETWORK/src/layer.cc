#include "../include/layer.hpp"

Layer::Layer(int previousLayerSize, int currentLayerSize)
{
    for(int i = 0; i < currentLayerSize; i++)
    {
        Neurons.push_back(new Neuron(previousLayerSize, currentLayerSize));
    }
}