#include "../headers/neuralnetwork.h"
#include "../ui/ui_neuralnetwork.h"
#include <QDebug>



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
}

NeuralNetwork::~NeuralNetwork()
{
    delete ui;
}

void NeuralNetwork::netTrain(){
    std::vector<int> specVector{10};
    net = new Network(specVector,
                      DPofThisWindow.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                      DPofThisWindow.GetCountsOfClasses(), 0.25);
    net->SetDataForTraining(DPofThisWindow.GetDataForTraining());
    net->SetDataForTesting(DPofThisWindow.GetDataForTesting());
    net->SetDataForValidation(DPofThisWindow.GetDataForValidation());
    net->Training(15);
    net->ValidationProduce();
}

void NeuralNetwork::on_StartToTrainNet_clicked() {
    std::thread t1(&NeuralNetwork::netTrain, this);
    t1.join();

    QMessageBox msb;
    msb.setWindowTitle("Net is ready!");
    msb.setText("Net is ready!");
    msb.exec();
    ui->Predict->setVisible(true);
}

void NeuralNetwork::on_Predict_clicked()
{
    qDebug() << net->TestProduce();
    ui->RealNumber->setText(QString::number(DPofThisWindow.GetDataForTraining().at(SpecVar)->GetLabel()));
    QImage im(DPofThisWindow.GetDataForTraining().at(SpecVar)->GetFeatureVector().data(), 28, 28, 28, QImage::Format_Indexed8);
    QImage temp = im.scaled(140, 140,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
    temp.invertPixels();
    ui->Picture->setPixmap(QPixmap::fromImage(temp));
    ui->PredictionOfNet->setText(QString::number(net->Prediction(DPofThisWindow.GetDataForTraining().at(SpecVar))));
    qDebug() << QString::number(net->Prediction(DPofThisWindow.GetDataForTraining().at(SpecVar)));
    SpecVar++;
}
