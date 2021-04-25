#include "../headers/mainwindow.h"
#include "../ui/ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    DP.ReadInputData("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-images.idx3-ubyte");
    DP.ReadInputLabel("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-labels.idx1-ubyte");
    this->setWindowTitle("Neural network!");
    ui->NetWindowButton->setText("Neural network window");
    ui->KMeansButton->setText("K-Means window");
    ui->KNNButton->setText("K-NN window");
}

MainWindow::~MainWindow() {
    delete ui;
}


void MainWindow::on_NetWindowButton_clicked() {
    QWidget *p = nullptr;
    NetWindow = new NeuralNetwork(p, &DP);
    NetWindow->open();
}

void MainWindow::on_KMeansButton_clicked() {
    QWidget *p = nullptr;
    KMeansWindow = new kmeans(p,&DP);
    KMeansWindow->open();
}

void MainWindow::on_KNNButton_clicked() {
    QWidget *p = nullptr;
    KNNWindow = new knn(p, &DP);
    KNNWindow->open();
}
