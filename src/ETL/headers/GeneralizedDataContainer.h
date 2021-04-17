//
// Created by 51btn on 20.02.2021.
//

#ifndef ML_GENERALIZEDDATACONTAINER_H
#define ML_GENERALIZEDDATACONTAINER_H
#include <utility>
#include <vector>
#include "Data.h"


//remember to initialize these vectors
class GeneralizedDataContainer {
public:
    GeneralizedDataContainer()= default;
    ~GeneralizedDataContainer(){
        for(Data * El : DataForTraining)
            delete El;
        DataForTraining.clear();
        for(Data *El : DataForTesting)
            delete El;
        DataForTesting.clear();
        for(Data *El : DataForValidation)
            delete El;
        DataForValidation.clear();
    }

    void SetDataForTesting(std::vector<Data *> SomeDataForTesting)       { DataForTesting = std::move(SomeDataForTesting);}
    void SetDataForValidation(std::vector<Data *> SomeDataForValidation) { DataForValidation = std::move(SomeDataForValidation);}
    void SetDataForTraining(std::vector<Data *> SomeDataForTraining)     { DataForTraining = std::move(SomeDataForTraining);}

protected:
    std::vector<Data *> DataForTraining;
    std::vector<Data *> DataForTesting;
    std::vector<Data *> DataForValidation;
};





#endif //ML_GENERALIZEDDATACONTAINER_H
