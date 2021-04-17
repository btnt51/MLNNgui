#include "../include/network.hpp"
#include "../include/layer.hpp"
#include "../../ETL/headers/DataProcessor.h"
#include <numeric>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate) {
    std::cout << spec.size();
    for(int i = 0; i < spec.size(); i++) {
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
    for(int i = 0; i < Weights.size() - 1; i++)
        Activation += Weights[i] * Inputs[i];
    return Activation;
}


double Network::Transfer(double Data, int Fashion){
    if(Fashion == 0)
        return 1.0 / (1.0 + exp(-Data));
    else
        return Data * (1 - Data);
}


std::vector<double> Network::ForwardPropagation(Data *data)
{
    /*std::vector<double> Inputs = data->GetNormalizedFeatureVector();

    for(auto &El : Layers) {
        std::vector<double> NewInputs;
        for(Neuron *Element : El->GetNeurons()) {
            double activation = this->ActivationEachNeuron(Element->GetWeights(), Inputs);
            Element->SetOutput(this->Transfer(activation));
            NewInputs.push_back(Element->GetOutput());
        }
        Inputs = NewInputs;
    }
    return Inputs; // output layer outputs*/
    std::vector<double> inputs = data->GetNormalizedFeatureVector();
    for(int i = 0; i < Layers.size(); i++)
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


void Network::BackPropagation(Data *data) {
    for(int i = static_cast<int>(Layers.size()) - 1; i >= 0; i--) {
        Layer *WorkingLayer = Layers.at(i);
        std::vector<double> errors;
        if(i != Layers.size() - 1) {
            for(int j = 0; j < WorkingLayer->GetNeurons().size(); j++) {
                double error = 0.0;
                for(Neuron *n : Layers.at(i + 1)->GetNeurons()) {
                    error += (n->Weights.at(j) * n->Delta);
                }
                errors.push_back(error);
            }

        } else {
            for(int j = 0; j < WorkingLayer->GetNeurons().size(); j++) {
                Neuron *n = WorkingLayer->GetNeurons().at(j);
                errors.push_back(static_cast<double >(data->GetClassVector().at(j) - n->Output)); // expected - actual
            }
        }

        for(int j = 0; j < WorkingLayer->GetNeurons().size(); j++) {
            Neuron *n = WorkingLayer->GetNeurons().at(j);
            n->SetDelta(errors.at(j) * this->Transfer(n->Output, 1)); //gradient / derivative part of back prop.
        }
    }
}


void Network::UpdateWeights(Data *data) {
    std::vector<double> inputs = data->GetNormalizedFeatureVector();
    for(int i = 0; i < Layers.size(); i++) {
        if(i != 0) {
            for(Neuron *n : Layers.at(i - 1)->GetNeurons()) {
                inputs.push_back(n->Output);
            }
        }

        for(Neuron *n : Layers.at(i)->GetNeurons()) {
            for(int j = 0; j < inputs.size(); j++) {
                n->Weights.at(j) += this->LearningRate * n->Delta * inputs.at(j);
            }
            n->Weights.back() += this->LearningRate * n->Delta;
        }
        inputs.clear();
    }
}


int Network::Prediction(Data * data) {
    std::vector<double> outputs = ForwardPropagation(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}


void Network::Training(int AmountOfEpochs) {
    for(int i = 0; i < AmountOfEpochs; i++) {
        double SumOfErrors = 0.0;
        for(Data *data : this->DataForTraining) {
            std::vector<double > Outputs = ForwardPropagation(data);
            std::vector<int > ExpectedResult = data->GetClassVector();
            double TempSumOfErrors = 0.0;
            for(int j = 0; j < Outputs.size(); j++)
                TempSumOfErrors += pow(static_cast<double >(ExpectedResult.at(j) - Outputs.at(j)), 2);
            SumOfErrors += TempSumOfErrors;
            BackPropagation(data);
            UpdateWeights(data);
        }

        printf("Iteration: %d \t Error=%.4f\n", i, SumOfErrors);
    }
}


double Network::TestProduce() {
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *data : this->DataForTesting) {
        count++;
        int index = Prediction(data);
        if(data->GetClassVector().at(index) == 1) numCorrect++;
    }

    TestPerformance = (numCorrect / count);
    return TestPerformance;
}


void Network::ValidationProduce() {
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *data : this->DataForValidation) {
        count++;
        int index = Prediction(data);
        if(data->GetClassVector().at(index) == 1) numCorrect++;
    }

    printf("ValidationProduce Performance: %.4f\n", numCorrect / count);
}
