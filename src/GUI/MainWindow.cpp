#include "MainWindow.hpp"

#include "Components.hpp"

#include <QApplication>
#include <QPushButton>


int gui::Run(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QPushButton hello("Hello world!");
    hello.resize(100, 30);

    hello.show();
    return app.exec();
}
