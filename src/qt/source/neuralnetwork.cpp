#include "../headers/neuralnetwork.h"
#include "../ui/ui_neuralnetwork.h"




NeuralNetwork::NeuralNetwork(QWidget *parent, DataProcessor *dp) :
    QDialog(parent),
    ui(new Ui::NeuralNetwork)
{
    ui->setupUi(this);
    DPofThisWindow = *dp;
    DPofThisWindow.CountClasses();
    DPofThisWindow.SplitData();
    ui->Predict->setVisible(false);
    this->setWindowTitle("Neural network");
    setWindowFlag(Qt::WindowContextHelpButtonHint,false);
    IsPushed = false;
}


NeuralNetwork::~NeuralNetwork()
{
    delete ui;
}

void NeuralNetwork::netTrain(){
    std::vector<int> specVector{10};
    Net = new Network(specVector,
                      DPofThisWindow.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                      DPofThisWindow.GetCountsOfClasses(), 0.25);
    Net->SetDataForTraining(DPofThisWindow.GetDataForTraining());
    Net->SetDataForTesting(DPofThisWindow.GetDataForTesting());
    Net->SetDataForValidation(DPofThisWindow.GetDataForValidation());
    Net->Training(15);
    Net->ValidationProduce();
    ui->PercentageOfErOfNet->setText(QString::number((1.0-Net->TestProduce())*100));
}

void NeuralNetwork::on_StartToTrainNet_clicked() {
    if(!IsPushed){
        IsPushed = true;
        std::thread t1(&NeuralNetwork::netTrain, this);
        t1.join();
        QMessageBox msb;
        msb.setWindowTitle("Net is ready!");
        msb.setText("Net is ready!");
        msb.exec();
        ui->Predict->setVisible(true);
    }
    else
        return;
}

void NeuralNetwork::on_Predict_clicked()
{
    ui->RealNumber->setText(QString::number(DPofThisWindow.GetDataForTraining().at(SpecVar)->GetLabel()));
    QImage im(DPofThisWindow.GetDataForTraining().at(SpecVar)->GetFeatureVector().data(), 28, 28, 28, QImage::Format_Indexed8);
    QImage temp = im.scaled(140, 140,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
    temp.invertPixels();
    ui->Picture->setPixmap(QPixmap::fromImage(temp));
    ui->PredictionOfNet->setText(QString::number(Net->Prediction(DPofThisWindow.GetDataForTraining().at(SpecVar))));
    SpecVar++;
}
