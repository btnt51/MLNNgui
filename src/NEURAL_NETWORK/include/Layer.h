#ifndef __LAYER_HPP
#define __LAYER_HPP
#include "Neuron.h"
#include <cstdint>
#include <vector>

class Layer {
public:
    Layer(int, int);
    std::vector<Neuron *> GetNeurons() { return Neurons; }
    std::vector<double >  GetLayerOutputs() { return LayerOutputs; }

private:
    std::vector<Neuron *> Neurons;
    std::vector<double> LayerOutputs;
};
#endif
