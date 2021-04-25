#ifndef KNN_H
#define KNN_H

#include <QDialog>
#include <QDebug>
#include "../ETL/headers/DataProcessor.h"
#include "../Algorithms/KNN/headers/KNN.h"

namespace Ui {
class knn;
}

class knn : public QDialog
{
    Q_OBJECT

public:
    explicit knn(QWidget *parent = nullptr,DataProcessor *dp = nullptr);
    ~knn();

private slots:
    void on_Predict_clicked();

    void on_StartToTrainAlg_clicked();

private:
    Ui::knn *ui;
    DataProcessor DP;
    KnnMethod *KNN;
    int SpecVar;
};

#endif // KNN_H
