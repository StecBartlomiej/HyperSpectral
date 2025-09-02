#include "ImageManager.hpp"

#include "ui_ImageManager.h"
#include "Image.hpp"

#include <QFileDialog>

extern Coordinator coordinator;

ImageManager::ImageManager(QWidget* parent): QWidget(parent), ui{new Ui::ImageManager()}
{
    ui->setupUi(this);

    ui->tableWidget->setColumnCount(3);
    ui->tableWidget->setHorizontalHeaderLabels({tr("Name"), tr("Path"), tr("Options")});
    ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);;
    ui->tableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);;


    connect(ui->FolderButton, &QPushButton::clicked, this, &ImageManager::FileLoadDialog);

    connect(ui->ClearButton, &QPushButton::clicked, this, &ImageManager::ClearTable);

}

void ImageManager::FileLoadDialog()
{
    QStringList files = QFileDialog::getOpenFileNames(
                    this,
                    tr("Select one or more files to open"),
                    directory_,
                    tr("ENVI HDR Files (*.hdr)"));

    if (files.empty())
        return;

    const auto start_length = selected_images_.size();
    directory_ = QFileInfo(files.front()).dir().absolutePath();

    for (const QString &envi_file: files)
    {
        const QFileInfo info(envi_file);
        const QString image_path = info.absolutePath() + "/" + info.baseName() + ".dat";

        LOG_INFO("FileLoadDialog: image_path={}, envi_path={}", image_path.toStdString(), envi_file.toStdString());

        Entity entity;
        try
        {
            entity = CreateImage(FilesystemPaths{envi_file.toStdString(), image_path.toStdString()});
            selected_images_.push_back(entity);
        }
        catch (const std::runtime_error &err)
        {
            LOG_ERROR("LoadDialog: caught exception '{}'  when creating image entity from envi path:{}.",
                err.what(), envi_file.toStdString());
            continue;
        }
        InsetImageToTable(entity);
        emit AddedNewImage(entity);
        LOG_INFO("LoadDialog: successfully loaded {}", envi_file.toStdString());
    }

    const auto diff_length = selected_images_.size() - start_length;
    LOG_INFO("LoadDialog: successfully created {} image entity, errors {}", diff_length, files.size() - diff_length);
}

void ImageManager::InsetImageToTable(Entity entity)
{
    const auto &paths = coordinator.GetComponent<FilesystemPaths>(entity);
    const auto file_name = QString::fromStdString( paths.envi_path.filename().string() );
    const auto file_path = QString::fromStdString( paths.envi_path.parent_path().string() );

    const auto row_count = ui->tableWidget->rowCount();

    ui->tableWidget->insertRow(row_count);
    ui->tableWidget->setItem(row_count, 0, new QTableWidgetItem(file_name));
    ui->tableWidget->setItem(row_count, 1, new QTableWidgetItem(file_path));

    auto *button = new QPushButton(tr("Delete"), ui->tableWidget);
    ui->tableWidget->setCellWidget(row_count, 2, button);

    connect(button, &QPushButton::clicked, this, [=] {
        const auto row = ui->tableWidget->indexAt(button->pos()).row();
        ui->tableWidget->removeRow(row);
        const auto map_node = row_to_entity_.extract(row);
        assert(!map_node.empty());
        emit DeletedImage(map_node.mapped());
    });
    row_to_entity_[row_count] = entity;
}

void ImageManager::ClearTable()
{
    ui->tableWidget->clearContents();
    ui->tableWidget->setRowCount(0);
}

