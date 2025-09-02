#ifndef IMAGEMANAGER_HPP
#define IMAGEMANAGER_HPP


#include <QWidget>
#include <EntityComponentSystem.hpp>


namespace Ui { class ImageManager; }


class ImageManager : public QWidget
{
    Q_OBJECT
public:
    ImageManager(QWidget *parent);

signals:
    void AddedNewImage(Entity entity);

    void DeletedImage(Entity entity);

protected slots:
    void FileLoadDialog();

private:
    void InsetImageToTable(Entity entity);

    void ClearTable();

private:
    Ui::ImageManager *ui;
    QString directory_ = "/home";
    std::vector<Entity> selected_images_;
    std::unordered_map<int, Entity> row_to_entity_;
};

#endif //IMAGEMANAGER_HPP
