#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "neuralnetwork.h"
#include "QBuffer"
#include "QImageReader"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_NetWindowButton_clicked();

private:
    Ui::MainWindow *ui;
    DataProcessor dp;
    Network *net;
    NeuralNetwork *netWindow;
};
#endif // MAINWINDOW_H
