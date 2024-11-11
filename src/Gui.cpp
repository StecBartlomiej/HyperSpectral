#include "Gui.hpp"
#include "Logger.hpp"
#include "Components.hpp"
#include "Image.hpp"

#include <cassert>
#include <map>


extern Coordinator coordinator;


[[nodiscard]] GLuint CreateTexture()
{
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    static constexpr GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glDisable(GL_TEXTURE_2D);

    return image_texture;
}

void DeleteTexture(GLuint texture)
{
    glDeleteTextures(1, &texture);
}

void LoadTextureGrayF32(GLuint texture,uint32_t width, uint32_t height, const float* data)
{
    assert(data != nullptr);

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, static_cast<GLsizei>(width), static_cast<GLsizei>(height),
                 0, GL_RED, GL_FLOAT, data);
    glDisable(GL_TEXTURE_2D);
}

GLFWwindow* CreateWindow()
{
    glfwSetErrorCallback(GlfwErrorCallback);
    if (!glfwInit())
    {
        LOG_CRITICAL("Failed to initialize GLFW");
        abort();
    }


    // GL 3.3 + GLSL 339
    const char* glsl_version = "#version 330 core";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "HyperCpp", nullptr, nullptr);
    if (window == nullptr)
    {
        LOG_CRITICAL("Failed to initialize GLFW window");
        abort();
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0)
    {
        LOG_ERROR("Failed to initialize OpenGL context");
        abort();
    }
    int major = GLAD_VERSION_MAJOR(version);
    int minor = GLAD_VERSION_MINOR(version);
    LOG_INFO("Successfully initialized OpenGL {}.{}", major, minor);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", 16.0f);

    return window;
}



GuiImageWindow* RegisterGuiImageWindow()
{
    auto *img_window_sys = coordinator.RegisterSystem<GuiImageWindow>();

    Attributes attributes{};
    attributes.set(coordinator.GetComponentType<FilesystemPaths>());
    attributes.set(coordinator.GetComponentType<ImageSize>());

    coordinator.SetSystemAttribute<GuiImageWindow>(attributes);

    return img_window_sys;
}


void GuiImageWindow::LoadImage()
{
    assert(entities_.contains(selected_));

    image_size_ = coordinator.GetComponent<ImageSize>(selected_);
    image_data_ = GetImageData(selected_);
    selected_band_ = 1;
}

void GuiImageWindow::LoadTexture()
{
    const auto band_offset = (selected_band_ - 1) * image_size_.width * image_size_.height;
    LoadTextureGrayF32(texture, image_size_.width, image_size_.height, image_data_.get() + band_offset);
}

void GuiImageWindow::Show()
{
    ImGui::Begin("Image");

    if (ImGui::BeginCombo("Obrazy", name_.c_str()))
    {
        for (const auto entity : entities_)
        {
            ImGui::PushID(reinterpret_cast<void*>(entity));
            auto name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();
            if (ImGui::Selectable(name.c_str(), entity == selected_))
            {
                selected_ = entity;
                name_ = name;
                selected_band_ = 1;
                LoadImage();
                LoadTexture();
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    if (ImGui::SliderInt("Dlugosc fali", &selected_band_, 1, image_size_.depth))
    {
        LoadTexture();
    }

    ImGui::Image((ImTextureID)(intptr_t)texture, ImVec2(image_size_.width * 2, image_size_.height * 2));

    ImGui::End();
}
