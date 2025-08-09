#ifndef MAINWINDOW_H
#define MAINWINDOW_H


#include <QMainWindow>


[[noreturn]]
int Run(int argc, char *argv[]);


class ViewImageWidget;

class MainWindow: public QMainWindow
{
public:
    MainWindow(QWidget *parent = nullptr);

private:
    ViewImageWidget *viewImageWidget_;
};

#endif //MAINWINDOW_H
