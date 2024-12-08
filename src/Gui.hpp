#ifndef GUI_HPP
#define GUI_HPP

#include "Components.hpp"
#include "EntityComponentSystem.hpp"
#include "Image.hpp"

#include <glad/gl.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>


[[nodiscard]] GLFWwindow* CreateWindow();

[[nodiscard]] GLuint CreateTexture();

void DeleteTexture(GLuint texture);

void LoadTextureGrayF32(GLuint texture,uint32_t width, uint32_t height, const float* data);

class GuiImage
{
public:
    void LoadImage(Entity entity);

    void LoadBandToTexture() const;

    ~GuiImage();

protected:
    Entity loaded_entity_{};
    GLuint texture_ = CreateTexture();
    ImageSize image_size_{0, 0, 1};
    std::shared_ptr<float[]> image_data_;
    int selected_band_ = 1;
};


class GuiImageWindow final: public System, public GuiImage
{
public:
    void Show();

private:
    std::string name_{};
};

[[nodiscard]] GuiImageWindow* RegisterGuiImageWindow();

void LoadOpenGLTexture(float *data, ImageSize size, GLuint texture, std::size_t selected_band);



class GuiThreshold final: public System, public GuiImage
{
public:
    void Show();

private:
    float threshold = 0.f;
    GLuint threshold_texture_ = CreateTexture();
    std::string name_{};
    bool show_threshold_ = false;
};

[[nodiscard]] GuiThreshold* RegisterGuiThreshold();

// Requires only filesystem paths
class PCAWindow final: public System, public GuiImage
{
public:
    void Show();

private:
    std::set<Entity> pca_entity_{};
    std::set<Entity> to_calculate_{};
    std::string name_{};
    ImageSize selected_size_{0, 0, 1};
};

[[nodiscard]] PCAWindow* RegisterPCAWindow();


[[nodiscard]] PCAWindow* RegisterPCAWindow();

struct HasNameSystem: public System
{
};

[[nodiscard]] HasNameSystem* RegisterHasNameSystem();


class Image
{
public:
    ~Image() { DeleteTexture(texture_); };

    void LoadImage(CpuMatrix matrix);

    void SetBand(uint32_t band);

    void Show(float x_scale = 1, float y_scale = 1) const;

    [[nodiscard]] ImageSize GetImageSize() const { return image_size_; }

    void Clear();

    [[nodiscard]] auto GetImageData() const { return image_data_; };

private:
    GLuint texture_ = CreateTexture();
    ImageSize image_size_{0, 0, 1};
    std::shared_ptr<float[]> image_data_;
    uint32_t selected_band_ = 1;
};


class ThresholdWindow final
{
public:
    void Show();

    void LoadEntity(Entity entity);

    void RunThreshold();

    [[nodiscard]] CpuMatrix GetThresholdMask() const;

    [[nodiscard]] auto LoadedEntity() const -> std::optional<Entity> { return loaded_entity_; }

private:
    std::optional<Entity> loaded_entity_{std::nullopt};
    Image original_img_;
    Image threshold_img_;
    ImageSize img_size_{0, 0, 1};
};


class MainWindow
{
public:
    MainWindow(HasNameSystem* has_name_system): has_name_system_{has_name_system} {}

    void Show();

private:
    ThresholdWindow threshold_window_{};
    std::string selected_img_name_ = "";
    HasNameSystem *has_name_system_;
};


#endif //GUI_HPP
