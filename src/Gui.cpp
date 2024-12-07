#include "Gui.hpp"
#include "Logger.hpp"
#include "Components.hpp"
#include "Image.hpp"

#include <cassert>
#include <map>
#include <chrono>


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


    ImFontGlyphRangesBuilder builder;
    builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
    builder.AddText(reinterpret_cast<const char *>(u8"Zażółć gęślą jaźń"));
    builder.AddText(reinterpret_cast<const char *>(u8"ZAŻÓŁĆ GĘŚLĄ JAŹŃ"));
    ImVector<ImWchar> ranges;
    builder.BuildRanges(&ranges);

    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", 16, nullptr, ranges.Data);
    io.Fonts->Build();

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


void LoadOpenGLTexture(float *data, ImageSize size, GLuint texture, std::size_t selected_band)
{
    const auto band_offset = (selected_band - 1) * size.width * size.height;
    LoadTextureGrayF32(texture, size.width, size.height, data + band_offset);
}


void GuiImage::LoadImage(Entity entity)
{
    loaded_entity_ = entity;

    auto cpu_matrix = GetImageData(loaded_entity_);
    assert(cpu_matrix.data != nullptr);

    image_data_ = cpu_matrix.data;
    image_size_ = cpu_matrix.size;

    selected_band_ = 1;
}

void GuiImage::LoadBandToTexture() const
{
    assert(image_data_ != nullptr);
    assert(selected_band_ <= image_size_.depth);

    LoadOpenGLTexture(image_data_.get(), image_size_, texture_, selected_band_);
}


GuiImage::~GuiImage()
{
    DeleteTexture(texture_);
}


void GuiImageWindow::Show()
{
    ImGui::Begin("Obraz");

    if (ImGui::BeginCombo("Obrazy", name_.c_str()))
    {
        for (const auto entity : entities_)
        {
            ImGui::PushID(reinterpret_cast<void*>(entity));
            auto name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();
            if (ImGui::Selectable(name.c_str(), entity == loaded_entity_))
            {
                name_ = name;
                selected_band_ = 1;
                LoadImage(entity);
                LoadBandToTexture();
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();

    if (ImGui::Button("Wybierz folder"))
    {
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        ImGui::OpenPopup(reinterpret_cast<const char *>(u8"Eksplorator plików"));
    }

    if (ImGui::BeginPopupModal(reinterpret_cast<const char *>(u8"Eksplorator plików"), NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("All those beautiful files will be deleted.\nThis operation cannot be undone!");
        ImGui::Separator();

        //static int unused_i = 0;
        //ImGui::Combo("Combo", &unused_i, "Delete\0Delete harder\0");

        static bool dont_ask_me_next_time = false;
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::Checkbox("Don't ask me next time", &dont_ask_me_next_time);
        ImGui::PopStyleVar();

        if (ImGui::Button("OK", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }

    if (ImGui::SliderInt(reinterpret_cast<const char *>(u8"Długość fali"), &selected_band_, 1, image_size_.depth))
    {
        LoadBandToTexture();
    }

    ImGui::Image((ImTextureID)(intptr_t)texture_, ImVec2(image_size_.width * 2, image_size_.height * 2));

    ImGui::End();
}

void GuiThreshold::Show()
{
    ImGui::Begin("Progowanie");

    if (ImGui::BeginCombo("Obraz", name_.c_str()))
    {
        for (const auto entity : entities_)
        {
            ImGui::PushID(reinterpret_cast<void*>(entity));
            auto name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();
            if (ImGui::Selectable(name.c_str(), entity == loaded_entity_))
            {
                name_ = name;
                selected_band_ = 1;
                LoadImage(entity);
                LoadBandToTexture();
                show_threshold_ = false;
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    ImGui::InputFloat(reinterpret_cast<const char *>(u8"Próg"), &threshold, 0.01f, 1.f);

    // ImGui::SameLine();


    if (ImGui::SliderInt(reinterpret_cast<const char *>(u8"Pasmo"), &selected_band_, 1, image_size_.depth))
    {
        LoadBandToTexture();
    }

    if (ImGui::Button("Uruchom progowanie", ImVec2(120, 0)) && !name_.empty())
    {
        const int band_offset = (selected_band_ - 1) * image_size_.width * image_size_.height;

        Matrix img{1, image_size_.height * image_size_.width, image_data_.get() + band_offset};

        auto [_size, data] = ManualThresholding(img, 0, threshold); // Passes only selected band so threshold on 0

        ImageSize band0 = {.width = image_size_.width, .height = image_size_.height, .depth = 1};
        LoadOpenGLTexture(data.get(), band0, threshold_texture_, 1);
        show_threshold_ = true;

    }

    ImGui::Image((ImTextureID)(intptr_t)texture_, ImVec2(image_size_.width * 2, image_size_.height * 2));

    if (show_threshold_)
    {
        ImGui::SameLine();
        ImGui::Image((ImTextureID)(intptr_t)threshold_texture_, ImVec2(image_size_.width * 2, image_size_.height * 2));
    }


    ImGui::End();
}

GuiThreshold* RegisterGuiThreshold()
{
    auto *window = coordinator.RegisterSystem<GuiThreshold>();

    Attributes attributes{};
    attributes.set(coordinator.GetComponentType<FilesystemPaths>());
    attributes.set(coordinator.GetComponentType<ImageSize>());

    coordinator.SetSystemAttribute<GuiThreshold>(attributes);

    return window;
}

void PCAWindow::Show()
{
    ImGui::Begin("PCA");


    if (ImGui::Button("Run PCA"))
    {
        auto get_data = [=, i=0]() mutable -> std::shared_ptr<float[]> {
            return GetImageData(*std::next(to_calculate_.begin(), i++)).data;
        };

        const auto height = selected_size_.width * selected_size_.height;
        const auto width = selected_size_.depth;

        pca_entity_ = to_calculate_;

        LOG_INFO("PCAWindow: started pca!");
        // TODO: run in std::async!
        // const auto pca_result = PCA(get_data, height, width, to_calculate_.size());
    }

    {
        static ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                       ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                       ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
        static int freeze_cols = 1;
        static int freeze_rows = 1;

        ImVec2 outer_size = ImVec2(0.0f, 16 * 8);
        if (ImGui::BeginTable("table_scrollx", 3, flags, outer_size))
        {
            ImGui::TableSetupScrollFreeze(freeze_cols, freeze_rows);
            ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_NoHide);
            ImGui::TableSetupColumn("Nazwa");
            ImGui::TableSetupColumn("Wybierz");
            ImGui::TableHeadersRow();
            for (int row = 0; row < entities_.size(); row++)
            {
                auto entity_iter = std::next(entities_.begin(), row);
                auto file_name = coordinator.GetComponent<FilesystemPaths>(*entity_iter).img_data.filename().string();

                ImGui::TableNextRow();

                if (!ImGui::TableSetColumnIndex(0))
                    continue;
                ImGui::Text("%d", row);

                if (!ImGui::TableSetColumnIndex(1))
                    continue;
                ImGui::Text(file_name.c_str());

                if (!ImGui::TableSetColumnIndex(2))
                    continue;

                bool is_enabled = to_calculate_.contains(*entity_iter);

                ImGui::PushID(row);
                if (ImGui::Checkbox("", &is_enabled) && is_enabled)
                {
                    to_calculate_.insert(*entity_iter);
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
    }

    if (ImGui::BeginCombo(reinterpret_cast<const char *>(u8"Wyświetl"), name_.c_str()))
    {
        for (const auto entity : pca_entity_)
        {
            ImGui::PushID(reinterpret_cast<void*>(entity));
            auto name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();
            if (ImGui::Selectable(name.c_str(), entity == loaded_entity_))
            {
                // TODO: add orginal image and projected from pca
                // selected_ = entity;
                // name_ = name;
                // selected_band_ = 1;
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::End();
}

PCAWindow* RegisterPCAWindow()
{
    auto *window = coordinator.RegisterSystem<PCAWindow>();

    Attributes attributes{};
    attributes.set(coordinator.GetComponentType<FilesystemPaths>());
    attributes.set(coordinator.GetComponentType<ImageSize>());

    coordinator.SetSystemAttribute<GuiImageWindow>(attributes);

    return window;
}

HasNameSystem* RegisterHasNameSystem()
{
    auto *sys = coordinator.RegisterSystem<HasNameSystem>();

    Attributes attributes{};
    attributes.set(coordinator.GetComponentType<FilesystemPaths>());

    coordinator.SetSystemAttribute<HasNameSystem>(attributes);

    return sys;
}


void Image::LoadImage(CpuMatrix matrix)
{
    assert(matrix.data != nullptr);

    image_data_ = matrix.data;
    image_size_ = matrix.size;

    selected_band_ = 1;

    LoadOpenGLTexture(image_data_.get(), image_size_, texture_, selected_band_);
}

void Image::SetBand(uint32_t band)
{
    assert(band >= 1 && band <= image_size_.depth);
    selected_band_ = band;
    LoadOpenGLTexture(image_data_.get(), image_size_, texture_, selected_band_);
}

void Image::Show(float x_scale, float y_scale) const
{
    ImGui::Image((ImTextureID)(intptr_t)texture_, ImVec2(image_size_.width * x_scale, image_size_.height * y_scale));
}

void Image::Clear()
{
    DeleteTexture(texture_);
    texture_ = CreateTexture();
}


void ThresholdWindow::Show()
{
    ImGui::Text("Skala");

    static int slider_value = 1;
    if (ImGui::SliderInt("Pasmo",  &slider_value, 1, img_size_.depth, "%d", ImGuiSliderFlags_ClampOnInput))
    {
        original_img_.SetBand(slider_value);
    }

    original_img_.Show();
    ImGui::SameLine(0, 10);
    threshold_img_.Show();

}

void ThresholdWindow::LoadEntity(Entity entity)
{
    auto cpu_matrix = GetImageData(entity);

    img_size_ = cpu_matrix.size;
    original_img_.LoadImage(cpu_matrix);
    threshold_img_.Clear();
    loaded_entity_ = entity;
}

void ThresholdWindow::RunThreshold()
{
    //
    const float threshold = 0.03;
    const std::size_t threshold_band = 1;

    const auto pixels_width = img_size_.width * img_size_.height;
    const auto data = original_img_.GetImageData().get() + pixels_width * threshold_band;

    Matrix img{.bands_height = 1, .pixels_width = pixels_width, .data = data};
    auto mask = ManualThresholding(img, 0, threshold);

    mask.size.height = img_size_.height;
    mask.size.width = img_size_.width;
    mask.size.depth = 1;

    threshold_img_.LoadImage(mask);
}

CpuMatrix ThresholdWindow::GetThresholdMask() const
{
    auto ptr = threshold_img_.GetImageData();

    ImageSize mask_size{.width = img_size_.width, .height = img_size_.height, .depth =  1};
    return {mask_size, std::move(ptr)};
}

void MainWindow::Show()
{
    ImGui::Begin("Ustawienia");

    ImGui::Button("Ustawienia");
    if (ImGui::BeginCombo("Wybierz obraz", selected_img_name_.c_str()))
    {
        for (const auto entity : has_name_system_->entities_)
        {
            ImGui::PushID(reinterpret_cast<void*>(entity));
            auto name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();

            if (ImGui::Selectable(name.c_str(), entity == threshold_window_.LoadedEntity().value_or(-1)))
            {
                selected_img_name_= name;
                threshold_window_.LoadEntity(entity);
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    if (ImGui::Button("Progowanie"))
    {

    }
    ImGui::SameLine();
    ImGui::Button("PCA");
    ImGui::SameLine();
    ImGui::Button("Klasyfikacja");
    ImGui::SameLine();
    if (ImGui::Button("Wszystko TEST"))
    {
        const auto start = std::chrono::high_resolution_clock::now();

        threshold_window_.RunThreshold();
        CpuMatrix mask = threshold_window_.GetThresholdMask();

        auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
        auto cpu_object = GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());

        auto LoadData = [&](std::size_t i) -> CpuMatrix { return cpu_object; };
        auto result_pca = PCA(LoadData,  cpu_img.size.depth, cpu_img.size.height * cpu_img.size.width, 1);

        std::ostringstream oss;
        for (std::size_t i = 0; i < result_pca.eigenvalues.size.height; ++i)
        {
            oss << result_pca.eigenvalues.data[i] << ", ";
        }
        LOG_INFO("PCA Result: {} eigenvalues: {}", result_pca.eigenvalues.size.height, oss.str());

        const auto end = std::chrono::high_resolution_clock::now();
        LOG_INFO("Time elapsed: {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        // Object Data
        // PCA ?
        // Classfiaiton

        // TODO:
        // 1. Cast image to pca dimensions
        // 2. Calculate statistical values
        // 3. Tree/SVM
    }


    ImGui::End();

    ImGui::Begin("Wyświetlanie");

    threshold_window_.Show();

    ImGui::End();
}



