#include "../headers/mainwindow.h"
#include "../ui/ui_mainwindow.h"
#include "QBuffer"
#include "QImageReader"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    dp.ReadInputData("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-images.idx3-ubyte");
    dp.ReadInputLabel("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-labels.idx1-ubyte");
    dp.SplitData();
    dp.CountClasses();
    ui->label_2->setText("All good");
    std::cout << "size of dp.GetDataForTraining " << dp.GetDataForTraining().size() << "\n";
    std::cout << " " << dp.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size() << " ";
    std::vector<int> hiddenLayers = {10};
    std::cout << hiddenLayers.size();
    Network net(hiddenLayers, dp.GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                    dp.GetCountsOfClasses(),
                    0.25);
    net.SetDataForTraining(dp.GetDataForTraining());
    net.SetDataForTesting(dp.GetDataForTesting());
    net.SetDataForValidation(dp.GetDataForValidation());
    net.Training(15);
    net.ValidationProduce();
    ui->label_2->setText(QString::number(net.TestProduce()));
   // QPixmap *p = new QPixmap(reinterpret_cast<const char *>(dp.GetDataForTraining().at(0)->GetFeatureVector().data()));
   // QByteArray img = QByteArray::fromRawData(reinterpret_cast<const char*>(dp.GetDataForTraining().at(0)->GetFeatureVector().data()),
     //                                              dp.GetDataForTraining().at(0)->GetFeatureVectorSize());

    //p.loadFromData(img);
    //QBuffer buffer(&img);
    //QImageReader reader(&buffer);
    //QImage image = reader.read();
   // ui->graphicsView->scene()-> addPixmap(QPixmap::fromImage(image));
    ui->label_4->setText(QString::number(dp.GetDataForTraining().at(3)->GetLabel()));
    QImage im(dp.GetDataForTraining().at(3)->GetFeatureVector().data(), 28, 28, 28, QImage::Format_Indexed8);
    QImage temp = /*im.scaledToWidth(168,Qt::SmoothTransformation);*/ im.scaled(140, 140,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
    //QImage temp1 = temp.scaled(280,280, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    ui->label_5->setPixmap(QPixmap::fromImage(temp));
    ui->label_6->setPixmap(QPixmap::fromImage(im));
}

MainWindow::~MainWindow()
{
    delete ui;
}

