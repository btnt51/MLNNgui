//
// Created by 51btn on 24.02.2021.
//
#include "../headers/Kmeans.h"
#define max std::numeric_limits<double>::max()


KMeansMethod::KMeansMethod(int NumberOfClusters) {
    this->NumberOfClusters = NumberOfClusters;
}


KMeansMethod::~KMeansMethod() {
    for(auto El : Clusters)
        delete El;
    Clusters.clear();
    UsedIndexes.clear();
}


void KMeansMethod::InitClusters() {
    Clusters.reserve(NumberOfClusters);
    for(int i=0; i < NumberOfClusters;i++){
        std::mt19937 Gen(time(nullptr)*DataForTraining.size());
        std::uniform_int_distribution<> UID(0, DataForTraining.size()-1);
        int Index = UID(Gen);
        while(UsedIndexes.find(Index) != UsedIndexes.end())
            Index = UID(Gen);

        //Clusters.push_back(new ClusterOfData(DataForTraining.at(Index)));
        Clusters.push_back(new Cluster(DataForTraining.at(Index)));
        UsedIndexes.insert(Index);
    }
}


void KMeansMethod::Train() {
    while (UsedIndexes.size() < DataForTraining.size()){
        std::mt19937 Gen(time(nullptr)*DataForTraining.size()*DataForValidation.size());
        std::uniform_int_distribution<> UID(0, DataForTraining.size()-1);
        int Index = UID(Gen);
        while(UsedIndexes.find(Index) != UsedIndexes.end())
            Index = UID(Gen);

        double MinimalDistance = max;
        int TheBestCluster{};
        for(int i = 0; i < static_cast<int>(Clusters.size()-1); ++i){
          double Distance = GetDistance(Clusters.at(i)->Centroid, DataForTraining.at(Index));
            if(Distance < MinimalDistance){
                MinimalDistance = Distance;
                TheBestCluster = i;
            }
        }
        Clusters.at(TheBestCluster)->AddToCluster(DataForTraining.at(Index));
        //Clusters.at(TheBestCluster)->AddToCluster(DataForTraining.at(Index));
        UsedIndexes.insert(Index);
    }
}


void KMeansMethod::InitClustersForEachClass() {
    std::unordered_set<int> ProcessedClasses;
    for(int i = 0; i < static_cast<int>(DataForTraining.size());i++){
        if(ProcessedClasses.find(DataForTraining.at(i)->GetLabel()) == ProcessedClasses.end()){
            //Clusters.push_back(new ClusterOfData (DataForTraining.at(i)));
            Clusters.push_back(new Cluster(DataForTraining.at(i)));
            UsedIndexes.insert(i);
            ProcessedClasses.insert(DataForTraining.at(i)->GetLabel());
        }
    }
}


double KMeansMethod::GetDistance(std::vector<double> *Centroid, Data *QueryPoint, int Fashion) {
    double Distance{};
    //int Dimensionality = Centroid->size();
    switch(Fashion)
    {
        default:
        {//Default method for finding distance in Euclid distance d(x,y)=sqrt((sigma((xi-yi)^2))/m)
            for(unsigned i = 0; i < Centroid->size()-1/*Dimensionality*/;++i)
                Distance += pow( Centroid->at(i) - QueryPoint->GetNormalizedFeatureVector().at(i),2);
            Distance /= Centroid->size()/*Dimensionality*/;
            return sqrt(Distance);
        }

        case 1: {
            //Manhattan distance by Minkowski metric d(x,y) = sigma(|xi-yi|)
            for (unsigned i = 0; i < Centroid->size()-1/*Dimensionality*/; ++i)
                Distance += std::abs(Centroid->at(i) - QueryPoint->GetNormalizedFeatureVector().at(i));
            return Distance;
        }

        case 2:{
            //Euclid distance by Minkowski metric d(x,y) = sqrt(sigma((xi - yi)^2))
            for(unsigned i = 0; i < Centroid->size()-1;++i)
                Distance += pow(Centroid->at(i) -  QueryPoint->GetNormalizedFeatureVector().at(i),2);
            return sqrt(Distance);
        }
    }
}


double KMeansMethod::ValidateProduce() {
    double CorrectedData{};
    for(auto &El : DataForValidation){
        double MinimalDistance = max;
        int Index{};

        for(int i = 0; i < static_cast<int>(Clusters.size());i++){
            double Distance = GetDistance(Clusters.at(i)->Centroid, El);
            if(Distance < MinimalDistance){
                MinimalDistance = Distance;
                Index = i;
            }

        }
        //std::cout << static_cast<int>(El->GetLabel()) << " -->" << Clusters.at(Index)->TheMostFrequentClass << "\n";
        if(Clusters.at(Index)->TheMostFrequentClassOfCluster == static_cast<int>(El->GetLabel()))
            CorrectedData++;
    }
    return 100.0 *(CorrectedData / static_cast<double>(DataForValidation.size()));
}


double KMeansMethod::TestProduce() {
    double CorrectedData{};
    for(auto &El : DataForTesting){
        double MinimalDistance = max;
        int Index{};

        for(int i = 0; i < static_cast<int>(Clusters.size());i++){
            double Distance = GetDistance(Clusters.at(i)->Centroid, El);
            if(Distance < MinimalDistance){
                MinimalDistance = Distance;
                Index = i;
            }

        }
        if(Clusters.at(Index)->TheMostFrequentClassOfCluster == El->GetLabel()) CorrectedData++;
    }
    return 100.0 *(CorrectedData / static_cast<double>(DataForTesting.size()));
}

int KMeansMethod::Predict(Data *El) {
    double MinimalDistance = max;
    int Index{};
    for(int i = 0; i < static_cast<int>(Clusters.size()); i++){
        double Distance = GetDistance(Clusters.at(i)->Centroid, El);
        if(Distance < MinimalDistance){
            MinimalDistance = Distance;
            Index = i;
        }

    }
    return static_cast<int>(Clusters.at(Index)->TheMostFrequentClassOfCluster);
}
