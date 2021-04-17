//
// Created by 51btn on 16.02.2021.
//

#include <headers/Data.h>

Data::Data() = default;

Data::~Data() {
    FeatureVector.clear();
    NormalizedVector.clear();
    ClassVector.clear();
}

void Data::SetClassVector(int Counts) {
    for (int i = 0; i < Counts;i++)
    {
        if(i == Label)
            ClassVector.push_back(1);
        else
            ClassVector.push_back(0);
    }
}

void Data::PrintFeatureVector() {
    std::cout << "[";
    for (uint8_t &El : this->FeatureVector)
        std::cout << El;
    std::cout << "]\n";
}

void Data::PrintNormalizedVector() {
    std::cout << NormalizedVector.size();
    std::cout << "[";
    for(double &El : NormalizedVector)
        std::cout << El << "\t";
    std::cout << "]\n";
}
