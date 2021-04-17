//
// Created by 51btn on 24.02.2021.
//
#ifndef ML_KNN_H
#define ML_KNN_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <headers/GeneralizedDataContainer.h>


class KnnMethod : protected GeneralizedDataContainer{
public:
    KnnMethod();
    KnnMethod(int);
    KnnMethod(int, std::vector<Data *> DT, std::vector<Data *> DTe, std::vector<Data*> DV);
    KnnMethod(std::vector<Data *> DT, std::vector<Data *> DTe, std::vector<Data*> DV);
    ~KnnMethod();


    void SetTheNumberOfNeighbors(int Number) { NumberOfNeighbors = Number; }
    void FindKNearest(Data *QueryPoint);

    int GetTheMostFrequentClass();
    double CalculateDistance(Data *QueryPoint, Data *Input, int Fashion = 0);
    double ValidateProduce();
    double TestProduce();

private:
    int NumberOfNeighbors{};
    std::vector<Data *> *Neighbors;
};
#endif //ML_KNN_H
