//
// Created by 51btn on 24.02.2021.
//
#include "../headers/KNN.h"

#define max std::numeric_limits<double>::max()

KnnMethod::KnnMethod() = default;


KnnMethod::KnnMethod(int Number) {
    NumberOfNeighbors = Number;
}


KnnMethod::KnnMethod(int Number, std::vector<Data *> DForTraining, std::vector<Data *> DForTesting,
                     std::vector<Data *> DForValidation) {
    NumberOfNeighbors = Number;
    DataForTraining = std::move(DForTraining);
    DataForTesting = std::move(DForTesting);
    DataForValidation = std::move(DForValidation);
}


KnnMethod::KnnMethod(std::vector<Data *> DForTraining, std::vector<Data *> DForTesting,
                     std::vector<Data *> DForValidation) {
    DataForTraining = std::move(DForTraining);
    DataForTesting = std::move(DForTesting);
    DataForValidation = std::move(DForValidation);
}


KnnMethod::~KnnMethod() {
    for(Data *El : *Neighbors)
        delete El;
    Neighbors->clear();
    delete Neighbors;
    Neighbors = nullptr;
}


void KnnMethod::FindKNearest(Data *QueryPoint) {
    Neighbors = new std::vector<Data *>();
    double Minimum = max;
    double PreviousMinimum = Minimum;
    int Index{};
    for(int i = 0; i < NumberOfNeighbors; i++){
        if(i == 0){
            for(int j = 0; j < DataForTraining.size(); j++) {
                double Distance = CalculateDistance(QueryPoint, DataForTraining.at(j));
                DataForTraining.at(j)->SetDistance(Distance);
                if (Distance < Minimum) {
                    Minimum = Distance;
                    Index = j;
                }
            }
            Neighbors->push_back(DataForTraining.at(Index));
            PreviousMinimum = Minimum;
            Minimum = max;

        } else {
            for(int j = 0; j < DataForTraining.size(); j++){
                double Distance = DataForTraining.at(j)->GetDistance();
                if(Distance > PreviousMinimum && Distance < Minimum){
                    Minimum = Distance;
                    Index = j;
                }
            }

            Neighbors->push_back(DataForTraining.at(Index));
            PreviousMinimum = Minimum;
            Minimum = max;
        }
    }
}


int KnnMethod::GetTheMostFrequentClass() {
    std::map<uint8_t, int> MapOfFrequency;
    for(auto &El : *Neighbors) {
        if (MapOfFrequency.find(El->GetLabel()) == MapOfFrequency.end())
            MapOfFrequency[El->GetLabel()] = 1;
        else
            MapOfFrequency[El->GetLabel()]++;
    }

    int Maximum{};
    int TheMost{};
    for(auto &El : MapOfFrequency){
        if(El.second > Maximum){
            Maximum = El.second;
            TheMost = static_cast<int>(El.first);
        }
    }
    return TheMost;
}


double KnnMethod::CalculateDistance(Data *QueryPoint, Data *Input, int Fashion) {
    double Distance{};
    int Dimensionality = QueryPoint->GetNormalizedFeatureVector().size();
    if(Dimensionality != Input->GetNormalizedFeatureVector().size()){
        std::cout << "Vector size mismatch.\n";
        exit(1);
    }

    switch(Fashion)
    {
        default:
        {
            //Default method for finding distance in Euclid distance d(x,y)=sqrt((sigma((xi-yi)^2))/m)
            for(unsigned i = 0; i < Dimensionality;++i)
                Distance += pow(QueryPoint->GetNormalizedFeatureVector().at(i) -
                        Input->GetNormalizedFeatureVector().at(i), 2);
            Distance /= Dimensionality;
            return sqrt(Distance);
        }
        case 1: {
            //Manhattan distance by Minkowski metric d(x,y) = sigma(|xi-yi|)
            for (unsigned i = 0; i < Dimensionality; ++i) {
                Distance += std::abs(
                        QueryPoint->GetNormalizedFeatureVector().at(i) - Input->GetNormalizedFeatureVector().at(i));
            }
            return Distance;
        }
        case 2:{
            //Euclid distance by Minkowski metric d(x,y) = sqrt(sigma((xi - yi)^2))
            for(unsigned i = 0; i < Dimensionality;++i){
                Distance += pow(QueryPoint->GetNormalizedFeatureVector().at(i)
                                - Input->GetNormalizedFeatureVector().at(i),2);
            }
            return sqrt(Distance);
        }
    }
}


double KnnMethod::ValidateProduce() {
    int Counter{};
    int DataIndex{};
    for(Data *El : DataForValidation) {
        FindKNearest(El);
        int Prediction = GetTheMostFrequentClass();
        DataIndex++;
        if(Prediction == El->GetLabel()) Counter++;
//        auto Percent = static_cast<double>((Counter*100.0)/DataIndex);
//        std::cout << "Current produce = " << Percent << "\n";

    }

    auto CurrentProduce = static_cast<double>((Counter * 100.0) / DataForValidation.size());
    std::cout << "ValidationProduce produce for Number of neighbors " << NumberOfNeighbors << " = " << CurrentProduce << "\n";
    return CurrentProduce;
}


double KnnMethod::TestProduce() {
    int Counter{};
    for(Data *El : DataForTesting){
        FindKNearest(El);
        int Prediction = GetTheMostFrequentClass();
        if(Prediction == El->GetLabel()) Counter++;
    }

    auto CurrentProduce = static_cast<double>((Counter * 100.0) / DataForTesting.size());
    std::cout << "Test produce for number of neighbors " << NumberOfNeighbors << " = " << CurrentProduce << "\n";
    return CurrentProduce;
}
