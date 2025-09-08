#ifndef MAINWINDOW_H
#define MAINWINDOW_H


#include <QMainWindow>

#include "ImageManager.hpp"
#include "ImageProcessing.hpp"


int Run(int argc, char *argv[]);


class ViewImageWidget;
class ImageManager;


class MainWindow: public QMainWindow
{
public:
    MainWindow(QWidget *parent = nullptr);

private:
    ViewImageWidget *viewImageWidget_;
    ImageManager *imageManager_;
    ImageProcessing *imageProcessing_;
};

#endif //MAINWINDOW_H
