#include "ImageProcessing.hpp"

#include "ui_ImageProcessing.h"

#include <QCheckBox>
#include <QScrollArea>


extern Coordinator coordinator;


ImageProcessing::ImageProcessing(QWidget* parent): QScrollArea(parent), ui_{new Ui::ImageProcessing()}
{
    widget_ = new QWidget(this);
    ui_->setupUi(widget_);

    this->setWidgetResizable(true);
    this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    this->setWidget(widget_);

    ui_->imageTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    ui_->imageTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);


}

void ImageProcessing::AddImage(Entity entity)
{
    InsertImageToTable(entity);
}

void ImageProcessing::SizeChanged(ImageSize size)
{
    assert(size.channel >= 1);
    ui_->pcaChannels->setValue(1);
    ui_->pcaChannels->setMaximum(static_cast<double>(size.channel));
}

void ImageProcessing::InsertImageToTable(Entity entity)
{
    const auto &paths = coordinator.GetComponent<FilesystemPaths>(entity);
    const auto &size = coordinator.GetComponent<ImageSize>(entity);

    const auto file_name = QString::fromStdString( paths.envi_path.filename().string() );

    const auto row_count = ui_->imageTable->rowCount();

    ui_->imageTable->insertRow(row_count);
    ui_->imageTable->setItem(row_count, 0, new QTableWidgetItem(file_name));
    ui_->imageTable->setItem(row_count, 1, new QTableWidgetItem(QString("%1x%2x%3").arg(size.width).arg(size.height).arg(size.channel)));

    auto *button = new QCheckBox(tr("Include"), ui_->imageTable);
    ui_->imageTable->setCellWidget(row_count, 2, button);

    // connect(button, &QPushButton::clicked, this, [=] {
    //     const auto row = ui->tableWidget->indexAt(button->pos()).row();
    //     ui->tableWidget->removeRow(row);
    //     const auto map_node = row_to_entity_.extract(row);
    //     assert(!map_node.empty());
    //     emit DeletedImage(map_node.mapped());
    // });
    // row_to_entity_[row_count] = entity;
}
