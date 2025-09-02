#include "MainWindow.hpp"

#include "Components.hpp"
#include "ViewImageWidget.hpp"
#include "ImageManager.hpp"

#include <QApplication>
#include <QTabWidget>


int Run(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow window{};
    window.resize(960, 720);

    window.show();

    return app.exec();
}

MainWindow::MainWindow(QWidget* parent):
            QMainWindow(parent),
            viewImageWidget_(new ViewImageWidget(this)),
            imageManager_(new ImageManager(this))
{
    QTabWidget *tabWidget = new QTabWidget(this);


    tabWidget->addTab(imageManager_, tr("Manger"));
    tabWidget->addTab(viewImageWidget_, tr("Viewer"));

    connect(imageManager_, &ImageManager::AddedNewImage, viewImageWidget_, &ViewImageWidget::AddImage);
    connect(imageManager_, &ImageManager::DeletedImage, viewImageWidget_, &ViewImageWidget::DeleteImage);

    setCentralWidget(tabWidget);
    setWindowTitle(tr("Hyperspectral"));
}
