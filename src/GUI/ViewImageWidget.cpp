#include "ViewImageWidget.hpp"

#include "ui_view_image.h"

#include <QSlider>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QApplication>
#include <fstream>

#include "EnviHeader.hpp"
#include "Image.hpp"


extern Coordinator coordinator;


ImageOpenGL::ImageOpenGL(QWidget* parent): QOpenGLWidget(parent), program(nullptr), texId(0)
{
    setMinimumSize(200,200);
}

ImageOpenGL::~ImageOpenGL()
{
    if (texId)
        glDeleteTextures(1, &texId);
    if (vbo)
        glDeleteBuffers(1, &vbo);
    delete program;
}

void ImageOpenGL::SetImage(CpuMatrix image)
{
    assert(image.size.width > 0 && image.size.height > 0);
    assert(image.data != nullptr);

    image_ = image;
    curr_channel = 0;
    update();

    emit NewImageSize(image.size);
}

void ImageOpenGL::initializeGL()
{
    initializeOpenGLFunctions();

    program = new QOpenGLShaderProgram(this);
    const char* vsrc = R"glsl(
            #version 330
            in vec2 a_pos;
            in vec2 a_uv;
            out vec2 v_uv;
            void main() {
                v_uv = a_uv;
                gl_Position = vec4(a_pos, 0.0, 1.0);
            }
        )glsl";
    const char* fsrc = R"glsl(
            #version 330
            uniform sampler2D u_tex;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                // texture() returns a vec4 where .r contains our float value
                float val = texture(u_tex, v_uv).r;
                fragColor = vec4(val, val, val, 1.0);
            }
        )glsl";
    program->addShaderFromSourceCode(QOpenGLShader::Vertex, vsrc);
    program->addShaderFromSourceCode(QOpenGLShader::Fragment, fsrc);
    program->bindAttributeLocation("a_pos", 0);
    program->bindAttributeLocation("a_uv", 1);
    program->link();

    static constexpr float verts[] = {
    //  pos.x   pos.y    u     v
        -1.0f, -1.0f,   0.0f, 1.0f,
         1.0f, -1.0f,   1.0f, 1.0f,
        -1.0f,  1.0f,   0.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 0.0f,
    };

    if (!vbo)
        glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (!texId)
        glGenTextures(1, &texId);

    // Swizzle so red channel appears in RGB
    static constexpr GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void ImageOpenGL::paintGL()
{
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);

    if (!program)
        return;

    program->bind();

    const auto *ptr_start = image_.data.get() + curr_channel * image_.size.width * image_.size.height;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, image_.size.width, image_.size.height, 0, GL_RED, GL_FLOAT, ptr_start);
    glActiveTexture(GL_TEXTURE0);

    program->setUniformValue("u_tex", 0);

    glBindTexture(GL_TEXTURE_2D, texId);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (const void*)(2 * sizeof(float)));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    program->release();
}

void ImageOpenGL::ChangeChannel(int new_channel)
{
    if (new_channel >= image_.size.channel || new_channel < 0)
    {
        LOG_ERROR("ImageOpenGL::ChangeChannel: Channel {} out of range from image with {} channels", new_channel, image_.size.channel);
        return;
    }
    curr_channel = new_channel;
    update();
}


void ImageOpenGL::LoadImage(Entity entity)
{
    const CpuMatrix img = GetImageData(entity);
    assert(img.data != nullptr);
    SetImage(img);
}

ViewImageWidget::ViewImageWidget(QWidget* parent): QWidget{parent}, ui{new Ui::HyperspectralViewImage()}
{
    ui->setupUi(this);

    connect(ui->horizontalSlider, &QSlider::valueChanged, ui->label, [=](int value) {
        ui->label->setText(tr("Kanał ") + QString::number(value));
    });

    connect(ui->horizontalSlider, &QSlider::valueChanged, ui->openGLWidget, &ImageOpenGL::ChangeChannel);
    connect(ui->openGLWidget, &ImageOpenGL::NewImageSize, ui->horizontalSlider, [=](ImageSize size) {
        ui->horizontalSlider->setRange(0, static_cast<int>(size.channel) - 1);
    });

    connect(ui->comboBox, &QComboBox::currentIndexChanged, ui->openGLWidget, [=](int index) {
        const QVariant v = ui->comboBox->itemData(index);
        const Entity entity = v.value<Entity>();
        ui->openGLWidget->LoadImage(entity);
    });
}

void ViewImageWidget::AddImage(Entity entity)
{
    const auto &paths = coordinator.GetComponent<FilesystemPaths>(entity);
    ui->comboBox->addItem(QString::fromStdString(paths.img_path.filename().string()), QVariant(entity));
}

void ViewImageWidget::DeleteImage(Entity entity)
{
    ui->comboBox->removeItem(entity);
}
