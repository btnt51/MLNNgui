#include "../headers/knn.h"
#include "../ui/ui_knn.h"

knn::knn(QWidget *parent, DataProcessor *dp) :
    QDialog(parent), ui(new Ui::knn) {
    ui->setupUi(this);
    DP = *dp;
    DP.SetDataPercent(0.005, 0.0025, 0.0025);
    DP.CountClasses();
    DP.SplitData();
    ui->Predict->setVisible(false);
    setWindowTitle("K-NN window");
    setWindowFlag(Qt::WindowContextHelpButtonHint,false);
    SpecVar = 0;
    IsPushed = false;
}

knn::~knn()
{
    delete ui;
}


void knn::on_StartToTrainAlg_clicked()
{
    if(!IsPushed){
        IsPushed = true;
        KNN = new KnnMethod(3, DP.GetDataForTraining(),
        DP.GetDataForTesting(), DP.GetDataForValidation());
        double Performance = 0, BestPerformance = 0;
        int BestK = 1;
        for(int k = 1; k <= 5; k++) {
            if(k == 1) {
                Performance = KNN->ValidateProduce();
                BestPerformance = Performance;
            } else {
                KNN->SetTheNumberOfNeighbors(k);
                Performance = KNN->ValidateProduce();
                if(Performance > BestPerformance) {
                    BestPerformance = Performance;
                    BestK = k;
                }
            }
        }

        KNN->SetTheNumberOfNeighbors(BestK);
        KNN->TestProduce();
        ui->Predict->setVisible(true);
    }
    else
        return;
    /*KNN = new KnnMethod(3, DP.GetDataForTraining(),
    DP.GetDataForTesting(), DP.GetDataForValidation());
    double Performance = 0, BestPerformance = 0;
    int BestK = 1;
    for(int k = 1; k <= 5; k++) {
        if(k == 1) {
            Performance = KNN->ValidateProduce();
            BestPerformance = Performance;
        } else {
            KNN->SetTheNumberOfNeighbors(k);
            Performance = KNN->ValidateProduce();
            if(Performance > BestPerformance) {
                BestPerformance = Performance;
                BestK = k;
            }
        }
    }

    KNN->SetTheNumberOfNeighbors(BestK);
    double per = KNN->TestProduce();
    ui->Predict->setVisible(true);
    ui->PercentageOfErOfKNN->setText(QString::number((1.0-per)*100));*/
}


void knn::on_Predict_clicked()
{
    ui->RealNumber->setText(QString::number(DP.GetDataForValidation().at(SpecVar)->GetLabel()));
    QImage im(DP.GetDataForValidation().at(SpecVar)->GetFeatureVector().data(), 28, 28, 28, QImage::Format_Indexed8);
    QImage temp = im.scaled(140, 140,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
    temp.invertPixels();
    ui->Picture->setPixmap(QPixmap::fromImage(temp));
    KNN->FindKNearest(DP.GetDataForValidation().at(SpecVar));
    ui->PredictionOfKNN->setText(QString::number(KNN->GetTheMostFrequentClass()));
    SpecVar++;
}


