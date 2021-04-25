#ifndef KMEANS_H
#define KMEANS_H

#include <QDialog>
#include <QDebug>
#include "../ETL/headers/DataProcessor.h"
#include "../Algorithms/K-means/headers/Kmeans.h"

namespace Ui {
class kmeans;
}

class kmeans : public QDialog
{
    Q_OBJECT

public:
    explicit kmeans(QWidget *parent = nullptr, DataProcessor *DP = nullptr);
    ~kmeans();

private slots:
    void on_StartToTrainAlg_clicked();

    void on_Predict_clicked();

private:
    Ui::kmeans *ui;
    DataProcessor DP;
    KMeansMethod *KMM;
    int SpecVar;

};

#endif // KMEANS_H
