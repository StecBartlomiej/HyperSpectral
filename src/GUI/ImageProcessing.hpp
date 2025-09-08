#ifndef IMAGEPROCESSING_HPP
#define IMAGEPROCESSING_HPP

#include "Components.hpp"

#include <QScrollArea>
#include <QWidget>


namespace Ui { class ImageProcessing; }


class ImageProcessing: public QScrollArea
{
    Q_OBJECT
public:
    ImageProcessing(QWidget *parent);

public slots:
    void AddImage(Entity entity);

    void SizeChanged(ImageSize size);

private:
    void InsertImageToTable(Entity entity);

private:
    Ui::ImageProcessing *ui_;
    QWidget *widget_;
};

#endif //IMAGEPROCESSING_HPP
