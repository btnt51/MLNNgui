//
// Created by 51btn on 24.02.2021.
//

#ifndef ML_KMEANS_H
#define ML_KMEANS_H

#include <limits>
#include <cmath>
#include <map>
#include <random>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <unordered_set>
#include <headers/GeneralizedDataContainer.h>
#include <headers/DataProcessor.h>

typedef struct ClusterOfData
{
    std::vector<double> *Centroid;
    std::vector<Data *> *ClusterOfPoints;
    std::map<int, int> CountsOfClasses;
    int TheMostFrequentClassOfCluster;
    ClusterOfData(Data *InitialPoint) {
        Centroid = new std::vector<double>;
        ClusterOfPoints = new std::vector<Data *>;
        for(auto El : InitialPoint->GetNormalizedFeatureVector()) {
            if(_isnan(El))
                Centroid->push_back(0);
            else
                Centroid->push_back(El);
        }

        ClusterOfPoints->push_back(InitialPoint);
        CountsOfClasses[InitialPoint->GetLabel()] = 1;
        TheMostFrequentClassOfCluster = InitialPoint->GetLabel();
    }

    void SetTheMostFrequentClass()
    {
        int PopularClass;
        int Frequince = 0;
        for(auto El : CountsOfClasses) {
            if(El.second > Frequince) {
                Frequince = El.second;
                PopularClass = El.first;
            }
        }
        TheMostFrequentClassOfCluster = PopularClass;
    }

    void AddToCluster(Data* Point) {
        int previous_size = ClusterOfPoints->size();
        ClusterOfPoints->push_back(Point);
        for(int i = 0; i < Centroid->size(); i++) {
            double El = Centroid->at(i);
            El *= previous_size;
            El += Point->GetNormalizedFeatureVector().at(i);
            El /= (double)ClusterOfPoints->size();
            Centroid->at(i) = El;
        }

        if(CountsOfClasses.find(Point->GetLabel()) == CountsOfClasses.end())
            CountsOfClasses[Point->GetLabel()] = 1;
        else
            CountsOfClasses[Point->GetLabel()]++;
        SetTheMostFrequentClass();
    }

} Cluster;


class KMeansMethod : public GeneralizedDataContainer {
public:
    KMeansMethod(int);
    ~KMeansMethod();

    void InitClusters();
    void InitClustersForEachClass();
    void Train();

    double GetDistance(std::vector<double> *, Data *, int Fashion = 2);
    double ValidateProduce();
    double TestProduce();
    int Predict(Data *El);
    std::vector<Cluster *> GetClusters() { return Clusters;}

private:
    int NumberOfClusters{};
    std::vector<Cluster *> Clusters;
    std::unordered_set<int> UsedIndexes;
};

#endif //ML_KMEANS_H
