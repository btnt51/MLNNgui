#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "../../ETL/headers/Data.h"
#include "Neuron.h"
#include "Layer.h"
#include "../../ETL/headers/GeneralizedDataContainer.h"

class Network : public GeneralizedDataContainer
{
  public:
    Network(std::vector<int> spec, int, int, double);
    ~Network();
    std::vector<double> ForwardPropagation(Data *data);
    double ActivationEachNeuron(std::vector<double> Weights, std::vector<double> Inputs); // dot product
    double Transfer(double Data, int Fashion = 0);
    double GetTestPerformance() { return this->TestPerformance;}

    int Prediction(Data *data); // return the index of the maximum value in the output array.
    double TestProduce();
    void ValidationProduce();
    void BackPropagation(Data *data);
    void UpdateWeights(Data *data);
    void Training(int AmountOfEpochs); // real Gym


private:
    std::vector<Layer *> Layers;
    double LearningRate;
    double TestPerformance;
};

#endif
