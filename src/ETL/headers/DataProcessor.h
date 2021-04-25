//
// Created by 51btn on 17.02.2021.
//
/*
 * Примерно затраченное время 4.5 часа без учёта комментирования
 */
#ifndef ML_DATAPROCESSOR_H
#define ML_DATAPROCESSOR_H

#include <string>
#include <map>
#include <unordered_set>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <thread>
#include <future>
#include "Data.h"

class DataProcessor {
public:
    DataProcessor();
    DataProcessor(double, double, double);
    ~DataProcessor();

    [[nodiscard]] int GetCountsOfClasses() const                     { return this->CountsOfClasses; }
    [[nodiscard]] int GetSizeOfArrayOfData() const                   { return this->ArrayOfData.size(); }
    [[nodiscard]] int GetSizeOfDataForTraining() const               { return this->DataForTraining.size(); }
    [[nodiscard]] int GetSizeOfDataForTesting() const                { return this->DataForTesting.size(); }
    [[nodiscard]] int GetSizeOfDataForValidation() const             { return this->DataForValidation.size(); }

    [[nodiscard]] std::vector<Data *> GetArrayOfData() const         { return this->ArrayOfData; }
    [[nodiscard]] std::vector<Data *> GetDataForTraining() const     { return this->DataForTraining; }
    [[nodiscard]] std::vector<Data *> GetDataForTesting() const      { return this->DataForTesting; }
    [[nodiscard]] std::vector<Data *> GetDataForValidation() const   { return this->DataForValidation; }

    static uint32_t CastData(const unsigned char *Bytes);

    void ReadInputData(const std::string&);
    void ReadInputLabel(const std::string&);
    void SplitData();
    void CountClasses();
    void NormalizeData(int AmountOfData);
    void SetDataPercent(double TP, double TE, double TV);
    void Print();

    void operator=(DataProcessor obj);

private:
    int FeatureVectorSize{};
    std::map<uint8_t, int> IntClass;
    std::map<std::string, int> StringClass;
    int CountsOfClasses{};
    std::vector<Data *> ArrayOfData{};
    std::vector<Data *> DataForTraining{};
    std::vector<Data *> DataForTesting{};
    std::vector<Data *> DataForValidation{};

    double TrainingPercent = 0.01;
    double TestPercent = 0.005;
    double ValidationPercent = 0.005;
};


#endif //ML_DATAPROCESSOR_H
