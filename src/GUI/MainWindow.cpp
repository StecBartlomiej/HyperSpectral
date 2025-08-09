#include "MainWindow.hpp"

#include "Components.hpp"
#include "ViewImageWidget.hpp"

#include <QApplication>
#include <QTabWidget>


int Run(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow window{};
    window.resize(800, 600);

    window.show();

    return app.exec();
}

MainWindow::MainWindow(QWidget* parent): QMainWindow(parent), viewImageWidget_(new ViewImageWidget(this))
{
    QTabWidget *tabWidget = new QTabWidget(this);


    tabWidget->addTab(viewImageWidget_, tr("Viewer"));

    setCentralWidget(tabWidget);
    setWindowTitle(tr("Hyperspectral"));
}
