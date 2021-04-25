//
// Created by 51btn on 17.02.2021.
//

#include <random>
#include <headers/DataProcessor.h>

DataProcessor::DataProcessor() {

}

DataProcessor::DataProcessor(double TP, double TeP, double VP){
    TrainingPercent = TP;
    TestPercent = TeP;
    ValidationPercent = VP;
}

DataProcessor::~DataProcessor() {
    for(Data *El : ArrayOfData){
        delete El;
    }
    ArrayOfData.clear();
    for(Data *El : DataForTraining){
        delete El;
    }
    DataForTraining.clear();
    for(Data *El : DataForTesting){
        delete El;
    }
    DataForTesting.clear();
    for(Data * El : DataForValidation){
        delete El;
    }
    DataForValidation.clear();
}

void DataProcessor::ReadInputData(const std::string& PathToFile) {
    uint32_t Header[4]{};     //Number | Number of Images | Number of Rows | Number of Columns
    unsigned char Bytes[4]{}; //using for read 32 bits of information for Header
    FILE *FileWithImages;     //file with data
    FileWithImages = std::fopen(PathToFile.c_str(), "rb");
    if(FileWithImages){
        for(uint32_t & El : Header){
            if(fread(Bytes, sizeof(Bytes), 1, FileWithImages)){
                El = CastData(Bytes);
            }
        }

   //     std::cout << Header[0] << "|" << Header[1] << "|" << Header[2] << "|"  << Header[3] << std::endl;
        uint32_t SizeOfImage = Header[2] * Header[3];
       // std::cout << "Header was read\n";
        for(int i = 0; i < Header[1];++i){
            Data *DataFromFile = new Data();
            uint8_t Pixel[1]{};

            for(int j = 0; j < SizeOfImage; ++j)
            {
                if(fread(Pixel, sizeof(Pixel), 1, FileWithImages))
                {
                    DataFromFile->AppendToFeatureVector(Pixel[0]);
                }
            }
            ArrayOfData.push_back(DataFromFile);
            ArrayOfData.back()->SetClassVector(CountsOfClasses);
        }

        NormalizeData(static_cast<int>(Header[1]));
        FeatureVectorSize = ArrayOfData.at(0)->GetFeatureVector().size();
       // std::cout << "All data was read\n";
    } else {
        std::cout << "There is some problem with the path to file with Images\n";
        exit(1);
    }
}

void DataProcessor::ReadInputLabel(const std::string& Path) {
    uint32_t Header[2]; // Number | Number of Images
    unsigned char Bytes[4];
    FILE *FileWithLabels;
    FileWithLabels = fopen(Path.c_str(),"rb");
    if(FileWithLabels){
        for(unsigned int & El : Header)
        {
            if(fread(Bytes,sizeof (Bytes), 1, FileWithLabels)){
                El = CastData(Bytes);
            }
        }

       // std::cout << "Labels headers was read\n";
        for(unsigned int i = 0; i < Header[1];++i){
            uint8_t Label[1];
            if(fread(Label, sizeof(Label), 1, FileWithLabels)) {
                ArrayOfData.at(i)->SetLabel(Label[0]);
            }
        }

     //   std::cout << "All labels was read\n";
    } else {
        std::cout << "There is a problem with the path to file with Labels\n";
        exit(1);
    }
}

void DataProcessor::SplitData() {
    int Counter{};
    int Index{};
    int SizeForTraining = static_cast<int>(GetSizeOfArrayOfData() * TrainingPercent);
    int SizeForTesting = static_cast<int>(GetSizeOfArrayOfData() * TestPercent);
    int SizeForValidation = static_cast<int>(GetSizeOfArrayOfData() * ValidationPercent);
    std::shuffle(ArrayOfData.begin(), ArrayOfData.end(), std::mt19937(std::random_device()()));

    //Splitting Training Data
    while(Counter < SizeForTraining){
        DataForTraining.push_back(ArrayOfData.at(Index++));
        Counter++;
    }
    Counter = 0;

    //Splitting Testing Data
    while (Counter < SizeForTesting){
        DataForTesting.push_back(ArrayOfData.at(Index++));
        Counter++;
    }
    Counter = 0;

    //Splitting ValidationProduce Data
    while(Counter < SizeForValidation){
        DataForValidation.push_back(ArrayOfData.at(Index++));
        Counter++;
    }
    /*std::cout << "There is data for training " << DataForTraining.size() << "." << std::endl <<
    "There is data for testing " << DataForTesting.size() << "."<< std::endl <<
    "There is data for validating " << DataForValidation.size() << "."<<std::endl;*/
}


void DataProcessor::CountClasses() {
    int Counter{};
    for(auto & El : ArrayOfData){
        if(IntClass.find(El->GetLabel()) == IntClass.end()){
            IntClass[El->GetLabel()] = Counter;
            El->SetEnumeratedLabel(Counter);
            Counter++;
        } else {
            El->SetEnumeratedLabel(IntClass[El->GetLabel()]);
        }
    }

    CountsOfClasses = Counter;
    for(Data *El : ArrayOfData)
        El->SetClassVector(CountsOfClasses);
   // std::cout << "Successfully extracted " << CountsOfClasses << " unique classes.\n";
}


void DataProcessor::NormalizeData(int AmountOfData) {
    std::vector<double> Minimum, Maximum;
    for(auto El : ArrayOfData.at(0)->GetFeatureVector()) {
        Minimum.push_back(El);
        Maximum.push_back(El);
    }

    for(int i = 1; i < ArrayOfData.size();i++) {
        auto El = ArrayOfData.at(i);
        for(int j = 1; j < El->GetFeatureVectorSize();j++){
            auto Value = static_cast<double>(El->GetFeatureVector().at(j));
            if(Value < Minimum.at(j)) Minimum.at(j) = Value;
            if(Value > Maximum.at(j)) Maximum.at(j) = Value;
        }
    }

    for(auto & El : ArrayOfData) {
        El->SetNormalizedFeatureVector(std::vector<double>());
        El->SetClassVector(CountsOfClasses);

        for(int j = 0; j < El->GetFeatureVectorSize(); j++) {
            if(Maximum[j] - Minimum[j] == 0)
                El->AppendToNormalizedFeatureVector(0.0);
            else {
                El->AppendToNormalizedFeatureVector(static_cast<double>(
                        (static_cast<double>(El->GetFeatureVector().at(j)) - Minimum[j])
                        / (Maximum[j] - Minimum[j])));
            }
        }
    }
}

void DataProcessor::SetDataPercent(double TP, double TEP, double VP){
    this->TrainingPercent = TP;
    this->TestPercent = TEP;
    this->ValidationPercent = VP;
}


uint32_t DataProcessor::CastData(const unsigned char *Bytes) {
    return (uint32_t)((Bytes[0] << 24) | (Bytes[1] << 16)| (Bytes[2] << 8) | (Bytes[3] << 0));
}


void DataProcessor::Print() {
    std::cout << "Training data:\n";
    for(Data *El : DataForTraining){
        for(auto Characteristic : El->GetNormalizedFeatureVector()){
            std::cout << Characteristic << " ";
        }
        std::cout << std::endl;
        std::cout << "--> " << El->GetLabel() << std::endl;
    }

    std::cout << "Data for tests\n";
    for(Data *El : DataForTesting){
        for(auto Characteristic : El->GetNormalizedFeatureVector()){
            std::cout << Characteristic << " ";
        }
        std::cout << std::endl;
        std::cout << "--> " << El->GetLabel() << std::endl;
    }

    std::cout << "Data for validation\n";
    for(Data *El : DataForValidation){
        for(auto Characteristic : El->GetNormalizedFeatureVector()){
            std::cout << Characteristic << " ";
        }
        std::cout << std::endl;
        std::cout << "--> " << El->GetLabel() << std::endl;
    }
}

void DataProcessor::operator=(DataProcessor obj)
{
    this->ArrayOfData = std::move(obj.ArrayOfData);
}


