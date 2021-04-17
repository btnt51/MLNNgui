#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron {
  public:
    Neuron(int, int);
    void InitializeWeights(int);
    void SetDelta(double Value) { Delta = Value; }
    void SetOutput(double Value) { Output = Value; }


    double Output;
    double Delta;
    std::vector<double> Weights;
};

#endif
