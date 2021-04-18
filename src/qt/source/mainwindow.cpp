#include "../headers/mainwindow.h"
#include "../ui/ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    dp.ReadInputData("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-images.idx3-ubyte");
    dp.ReadInputLabel("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-labels.idx1-ubyte");
    ui->label_2->setText("All good");
    std::vector<int> hiddenLayers = {10};
    std::cout << hiddenLayers.size();
    /*net = new Network(hiddenLayers, dp.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                    dp.GetCountsOfClasses(),
                    0.25);
    net->SetDataForTraining(dp.GetDataForTraining());
    net->SetDataForTesting(dp.GetDataForTesting());
    net->SetDataForValidation(dp.GetDataForValidation());
    net->Training(15);
    net->ValidationProduce();
    ui->label_2->setText(QString::number(net->TestProduce()));*/
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    static int i = 0;
    ui->label_4->setText(QString::number(dp.GetDataForTraining().at(i)->GetLabel()));
    QImage im(dp.GetDataForTraining().at(i)->GetFeatureVector().data(), 28, 28, 28, QImage::Format_Indexed8);
    QImage temp = im.scaled(140, 140,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
    temp.invertPixels();
    ui->label_5->setPixmap(QPixmap::fromImage(temp));
    ui->label_6->setPixmap(QPixmap::fromImage(im));
    ui->label_8->setText(QString::number(net->Prediction(dp.GetDataForTraining().at(i))));
    i++;
}

void MainWindow::on_pushButton_2_clicked()
{
    QWidget *p = nullptr;
    netWindow = new NeuralNetwork(p, &dp);
    netWindow->open();
}
