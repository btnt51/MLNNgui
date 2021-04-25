#include "../headers/kmeans.h"
#include "../ui/ui_kmeans.h"

kmeans::kmeans(QWidget *parent, DataProcessor *dp) :
    QDialog(parent), ui(new Ui::kmeans) {
    ui->setupUi(this);
    DP = *dp;
    DP.CountClasses();
    DP.SplitData();
    SpecVar = 0;
    ui->Predict->setVisible(false);
    setWindowTitle("K-Means windows");
    setWindowFlag(Qt::WindowContextHelpButtonHint,false);
    IsPushed = false;
}


kmeans::~kmeans() {
    delete ui;
}


void kmeans::on_StartToTrainAlg_clicked() {
    if(!IsPushed){
        IsPushed = true;
        int BestK = 0,Performance = 0, BestPerformance = 0;
        for(int k = 1; k < 30; k++) {
               auto *km = new KMeansMethod(k);
               km->SetDataForTraining(DP.GetDataForTraining());
               km->SetDataForTesting(DP.GetDataForTesting());
               km->SetDataForValidation(DP.GetDataForValidation());
               km->InitClusters();
               km->Train();
               //qDebug() << km->ValidateProduce();
               Performance = km->ValidateProduce();
               if(Performance > BestPerformance) {
                   BestPerformance = Performance;
                   BestK = k;
               }
           }
           KMM = new KMeansMethod(BestK);
           KMM->SetDataForTraining(DP.GetDataForTraining());
           KMM->SetDataForTesting(DP.GetDataForTesting());
           KMM->SetDataForValidation(DP.GetDataForValidation());
           KMM->InitClusters();
           KMM->Train();
           ui->Predict->setVisible(true);
    } else
        return;
}


void kmeans::on_Predict_clicked() {
    ui->RealNumber->setText(QString::number(DP.GetDataForValidation().at(SpecVar)->GetLabel()));
    QImage im(DP.GetDataForValidation().at(SpecVar)->GetFeatureVector().data(), 28, 28, 28, QImage::Format_Indexed8);
    QImage temp = im.scaled(140, 140,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
    temp.invertPixels();
    ui->Picture->setPixmap(QPixmap::fromImage(temp));
    ui->PredictionOfKMeans->setText(QString::number(KMM->Predict(DP.GetDataForValidation().at(SpecVar))));
    SpecVar++;
}
