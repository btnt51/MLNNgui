#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <QDialog>
#include <QMessageBox>
#include "../../ETL/headers/DataProcessor.h"
#include "../../NEURAL_NETWORK/include/network.hpp"

namespace Ui {
class NeuralNetwork;
}

class NeuralNetwork : public QDialog
{
    Q_OBJECT

public:
    explicit NeuralNetwork(QWidget *parent = nullptr, DataProcessor *dp = nullptr);
    ~NeuralNetwork();

    void netTrain(std::vector<int>);

private slots:
    void on_StartToTrainNet_clicked();

    void on_Predict_clicked();

private:
    Ui::NeuralNetwork *ui;
    DataProcessor DPofThisWindow;
    Network *net;
    int SpecVar = 0;
};

#endif // NEURALNETWORK_H
