#ifndef GUI_HPP
#define GUI_HPP

#include "Components.hpp"
#include "EntityComponentSystem.hpp"

#include <glad/gl.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>


[[nodiscard]] GLFWwindow* CreateWindow();

[[nodiscard]] GLuint CreateTexture();

void DeleteTexture(GLuint texture);

void LoadTextureGrayF32(GLuint texture,uint32_t width, uint32_t height, const float* data);

[[nodiscard]] std::shared_ptr<float> GetImageData(Entity entity);


class GuiImageWindow final: public System
{
public:
    void Show();

    void LoadImage();

    void LoadTexture();

private:
    Entity selected_{};
    GLuint texture = CreateTexture();
    int selected_band_ = 1;
    std::string name_{};
    ImageSize image_size_{0, 0, 1};
    std::shared_ptr<float> image_data_;
};

[[nodiscard]] GuiImageWindow* RegisterGuiImageWindow();


#endif //GUI_HPP
