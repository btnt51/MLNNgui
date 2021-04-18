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
}

NeuralNetwork::~NeuralNetwork()
{
    delete ui;
}

void NeuralNetwork::netTrain(std::vector<int> specVector){
    qDebug() <<
                DPofThisWindow.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size();
    net = new Network(specVector,
                      DPofThisWindow.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                      DPofThisWindow.GetCountsOfClasses(), 0.25);
    net->SetDataForTraining(DPofThisWindow.GetDataForTraining());
    net->SetDataForTesting(DPofThisWindow.GetDataForTesting());
    net->SetDataForValidation(DPofThisWindow.GetDataForValidation());
    net->Training(15);
    net->ValidationProduce();
}

void NeuralNetwork::on_StartToTrainNet_clicked()
{
    std::vector<int> specVector;
    QString SpecString = ui->AmountOfNeurons->text();
    SpecString.append(" ");
    auto l = SpecString.indexOf(" ");
    std::vector<int> temp;
    QString tempstr;
    int k = 0;
    while(k < ui->AmountOfLayers->text().toInt()){
        for(int i = 0; i < l; i++){
            tempstr.append(SpecString[i]);
        }
        k++;
        SpecString.remove(0,l+1);
        temp.push_back(tempstr.toInt());
        l = SpecString.indexOf(" ");
        tempstr.clear();
    }
    qDebug() << temp.at(0);
    for(int i = 0; i < ui->AmountOfLayers->text().toInt(); i++){
        specVector.push_back(temp.at(i));
    }
    //netTrain(specVector);
    std::thread t1(&NeuralNetwork::netTrain, this, specVector);
    t1.join();
    qDebug() << "3" << ui->LearningRate->text().toDouble();

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
