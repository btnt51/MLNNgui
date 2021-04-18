#include "../include/network.hpp"
#include "../include/layer.hpp"
#include "../../ETL/headers/DataProcessor.h"
#include <numeric>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate) {
    std::cout << spec.size();
    for(int i = 0; i < static_cast<int>(spec.size()); i++) {
        if(i == 0)
            Layers.push_back(new Layer(inputSize, spec.at(i)));
        else
            Layers.push_back(new Layer(Layers.at(i - 1)->GetNeurons().size(), spec.at(i)));
    }
    Layers.push_back(new Layer(Layers.at(Layers.size()-1)->GetNeurons().size(), numClasses));
    this->LearningRate = learningRate;
}

Network::~Network() {}

double Network::ActivationEachNeuron(std::vector<double> Weights, std::vector<double> Inputs) {
    double Activation = Weights.back(); // bias term
    for(int i = 0; i < static_cast<int>(Weights.size() - 1); i++)
        Activation += Weights[i] * Inputs[i];
    return Activation;
}


double Network::Transfer(double Data, int Fashion){
    if(Fashion == 0)
        return 1.0 / (1.0 + exp(-Data));
    else
        return Data * (1 - Data);
}


std::vector<double> Network::ForwardPropagation(Data *Data)
{
    std::vector<double> inputs = Data->GetNormalizedFeatureVector();
    for(int i = 0; i < static_cast<int>(Layers.size()); i++)
    {
        Layer *layer = Layers.at(i);
        std::vector<double> newInputs;
        for(Neuron *n : layer->GetNeurons())
        {
            double activation = this->ActivationEachNeuron(n->Weights, inputs);
            n->SetOutput(this->Transfer(activation));
            newInputs.push_back(n->Output);
        }
        inputs = newInputs;
    }
    return inputs; // output layer outputs
}


void Network::BackPropagation(Data *Data) {
    for(int i = static_cast<int>(Layers.size()) - 1; i >= 0; i--) {
        Layer *WorkingLayer = Layers.at(i);
        std::vector<double> errors;
        if(i != static_cast<int>(Layers.size() - 1)) {
            for(int j = 0; j < static_cast<int>(WorkingLayer->GetNeurons().size()); j++) {
                double error = 0.0;
                for(Neuron *n : Layers.at(i + 1)->GetNeurons()) {
                    error += (n->Weights.at(j) * n->Delta);
                }
                errors.push_back(error);
            }

        } else {
            for(int j = 0; j < static_cast<int>(WorkingLayer->GetNeurons().size()); j++) {
                Neuron *n = WorkingLayer->GetNeurons().at(j);
                errors.push_back(static_cast<double >(Data->GetClassVector().at(j) - n->Output)); // expected - actual
            }
        }

        for(int j = 0; j < static_cast<int>(WorkingLayer->GetNeurons().size()); j++) {
            Neuron *n = WorkingLayer->GetNeurons().at(j);
            n->SetDelta(errors.at(j) * this->Transfer(n->Output, 1)); //gradient / derivative part of back prop.
        }
    }
}


void Network::UpdateWeights(Data *Data) {
    std::vector<double> inputs = Data->GetNormalizedFeatureVector();
    for(int i = 0; i < static_cast<int>(Layers.size()); i++) {
        if(i != 0) {
            for(Neuron *n : Layers.at(i - 1)->GetNeurons()) {
                inputs.push_back(n->Output);
            }
        }

        for(Neuron *n : Layers.at(i)->GetNeurons()) {
            for(int j = 0; j < static_cast<int>(inputs.size()); j++) {
                n->Weights.at(j) += this->LearningRate * n->Delta * inputs.at(j);
            }
            n->Weights.back() += this->LearningRate * n->Delta;
        }
        inputs.clear();
    }
}


int Network::Prediction(Data * Data) {
    std::vector<double> outputs = ForwardPropagation(Data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}


void Network::Training(int AmountOfEpochs) {
    for(int i = 0; i < AmountOfEpochs; i++) {
        double SumOfErrors = 0.0;
        for(Data *Data : this->DataForTraining) {
            std::vector<double > Outputs = ForwardPropagation(Data);
            std::vector<int > ExpectedResult = Data->GetClassVector();
            double TempSumOfErrors = 0.0;
            for(int j = 0; j < static_cast<int>(Outputs.size()); j++)
                TempSumOfErrors += pow(static_cast<double >(ExpectedResult.at(j) - Outputs.at(j)), 2);
            SumOfErrors += TempSumOfErrors;
            BackPropagation(Data);
            UpdateWeights(Data);
        }

        printf("Iteration: %d \t Error=%.4f\n", i, SumOfErrors);
    }
}


double Network::TestProduce() {
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *Data : this->DataForTesting) {
        count++;
        int index = Prediction(Data);
        if(Data->GetClassVector().at(index) == 1) numCorrect++;
    }

    TestPerformance = (numCorrect / count);
    return TestPerformance;
}


void Network::ValidationProduce() {
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *Data : this->DataForValidation) {
        count++;
        int index = Prediction(Data);
        if(Data->GetClassVector().at(index) == 1) numCorrect++;
    }

    printf("ValidationProduce Performance: %.4f\n", numCorrect / count);
}
