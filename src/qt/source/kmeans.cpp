#include "../headers/kmeans.h"
#include "../ui/ui_kmeans.h"

kmeans::kmeans(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::kmeans)
{
    ui->setupUi(this);
}

kmeans::~kmeans()
{
    delete ui;
}

void kmeans::on_StartToTrainAlg_clicked()
{

}

void kmeans::on_Predict_clicked()
{

}
