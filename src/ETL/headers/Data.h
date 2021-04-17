//
// Created by 51btn on 16.02.2021.
//
/*
 * Примерно затраченное время 4 часа, без учета комментирования кода
 */

#ifndef ML_DATA_H
#define ML_DATA_H


#include <utility>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <iostream>

class Data{
public:
    Data();
    ~Data();
    [[nodiscard]] double GetDistance() const                      { return this->Distance; }
    int GetFeatureVectorSize()                                    { return this->FeatureVector.size(); }
    [[nodiscard]] uint8_t GetLabel() const                        { return this->Label; }
    [[nodiscard]] uint8_t GetEnumeratedLabel() const              { return this->EnumeratedLabel; }

    std::vector<uint8_t> GetFeatureVector()                       { return this->FeatureVector; }
    std::vector<double> GetNormalizedFeatureVector()              { return this->NormalizedVector; }
    std::vector<int> GetClassVector()                             { return this->ClassVector; }

    void SetDistance(double Value)                                { this->Distance = Value; }
    void SetFeatureVector(std::vector<uint8_t> Vec)               { this->FeatureVector = std::move(Vec); }
    void SetNormalizedFeatureVector(std::vector<double> Vec)      { this->NormalizedVector = std::move(Vec); }
    void SetLabel(uint8_t Lab)                                    { this->Label = Lab; }
    void SetEnumeratedLabel(uint8_t EnumLab)                      { this->EnumeratedLabel = EnumLab; }
    void AppendToFeatureVector(uint8_t Value)                     { FeatureVector.push_back(Value); }
    void AppendToNormalizedFeatureVector(double Value)            { NormalizedVector.push_back(Value); }
    void SetClassVector(int Counts);
    void PrintFeatureVector();
    void PrintNormalizedVector();

private:
    uint8_t Label{};                        //handling info about something (like what is number on photo)
    uint8_t EnumeratedLabel{};
    double Distance{};

    std::vector<uint8_t> FeatureVector;     //containing unprepared tokens or pixels
    std::vector<double> NormalizedVector;   //handling prepared tokens or pixels
    std::vector<int> ClassVector;
};


#endif //ML_DATA_H
