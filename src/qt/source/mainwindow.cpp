#include "../headers/mainwindow.h"
#include "../ui/ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    dp.ReadInputData("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-images.idx3-ubyte");
    dp.ReadInputLabel("D:\\c++\\guimlnn\\MLNNgui\\dataset\\train-labels.idx1-ubyte");
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
    netWindow = new NeuralNetwork(p, &dp);
    netWindow->open();
}

void MainWindow::on_KMeansButton_clicked() {

}

void MainWindow::on_KNNButton_clicked() {

}
