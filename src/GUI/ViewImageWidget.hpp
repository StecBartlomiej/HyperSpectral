#ifndef VIEWIMAGEWIDGET_HPP
#define VIEWIMAGEWIDGET_HPP

#include "EntityComponentSystem.hpp"
#include "Image.hpp"

#include <QOpenGLWidget>
#include <QOpenGLShader>
#include <QOpenGLFunctions>


namespace Ui { class HyperspectralViewImage; }

class QOpenGLTexture;



class ImageOpenGL: public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    ImageOpenGL(QWidget* parent = nullptr);

    ~ImageOpenGL() override;

    void SetImage(const CpuMatrix& image);

signals:
    void NewImageSize(ImageSize img);

public slots:
    void LoadImage(Entity entity);

    void ChangeChannel(int new_channel);

protected:
    void initializeGL() override;

    void paintGL() override;

private:
    QOpenGLShaderProgram* program;
    CpuMatrix image_;
    std::size_t curr_channel = 0;
    GLuint texId = 0;
    GLuint vbo = 0;
};



class ViewImageWidget : public QWidget
{
public:
    ViewImageWidget(QWidget *parent);

private:
    Ui::HyperspectralViewImage *ui;
};


#endif //VIEWIMAGEWIDGET_HPP
