#include "Gui.hpp"
#include "Logger.hpp"
#include "Components.hpp"
#include "Image.hpp"

#include <cassert>
#include <map>
#include <chrono>
#include <Classification.hpp>


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


void ImageViewWindow::Show()
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

void ImageViewWindow::LoadEntity(Entity entity)
{
    auto cpu_matrix = GetImageData(entity);

    img_size_ = cpu_matrix.size;
    original_img_.LoadImage(cpu_matrix);
    threshold_img_.Clear();
    loaded_entity_ = entity;
}

void ImageViewWindow::RunThreshold(float threshold, std::size_t threshold_band)
{
    const auto pixels_width = img_size_.width * img_size_.height;
    const auto data = original_img_.GetImageData().get() + pixels_width * (threshold_band - 1);

    Matrix img{.bands_height = 1, .pixels_width = pixels_width, .data = data};
    auto mask = ManualThresholding(img, 0, threshold);

    mask.size.height = img_size_.height;
    mask.size.width = img_size_.width;
    mask.size.depth = 1;

    threshold_img_.LoadImage(mask);
}

CpuMatrix ImageViewWindow::GetThresholdMask() const
{
    auto ptr = threshold_img_.GetImageData();

    ImageSize mask_size{.width = img_size_.width, .height = img_size_.height, .depth =  1};
    return {mask_size, std::move(ptr)};
}

void TransformedImageWindow::Show()
{
    ImGui::SeparatorText("Wynik PCA");
    if (ImGui::SliderInt("Pasmo##PCA_PASMO",  &slider_value_, 1, img_size_.depth, "%d", ImGuiSliderFlags_ClampOnInput))
    {
        original_img_.SetBand(img_size_.depth + 1 - slider_value_);
    }
    original_img_.Show();
}

void TransformedImageWindow::Load(const CpuMatrix &cpu_matrix)
{
    img_size_ = cpu_matrix.size;

    original_img_.LoadImage(cpu_matrix);
    original_img_.SetBand(img_size_.depth);
    slider_value_ = 1;
}

void ThresholdPopupWindow::RunThreshold()
{
    const auto pixels_width = img_size_.width * img_size_.height;
    const auto data = original_img_.GetImageData().get() + pixels_width * (selected_band_ - 1);

    Matrix img{.bands_height = 1, .pixels_width = pixels_width, .data = data};
    auto mask = ManualThresholding(img, 0, threshold_value_);

    mask.size.height = img_size_.height;
    mask.size.width = img_size_.width;
    mask.size.depth = 1;

    threshold_img_.LoadImage(mask);
}

void DataInputImageWindow::Show()
{
    if (ImGui::BeginCombo(reinterpret_cast<const char*>(u8"Plik nagłówkowy ENVI"),
        envi_header_path_.filename().string().c_str()))
    {
        std::size_t idx = 0;
        for (const auto& filepath: std::filesystem::directory_iterator(loading_image_path_))
        {
            if (!filepath.is_regular_file() || filepath.path().extension() != ".hdr")
                continue;

            ImGui::PushID(reinterpret_cast<void*>(idx));
            const auto name = filepath.path().filename().string();

            bool selected = false;
            if (ImGui::Selectable(name.c_str(), selected))
            {
                envi_header_path_ = filepath.path();
            }
            ImGui::PopID();
            ++idx;
        }
        ImGui::EndCombo();
    }

    constexpr static ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
    static int freeze_cols = 1;
    static int freeze_rows = 1;

    ImVec2 outer_size = ImVec2(0.0f, 16 * 8);
    if (ImGui::BeginTable(reinterpret_cast<const char*>(u8"Wybierz dane wejściowe"), 3, flags, outer_size))
    {
        ImGui::TableSetupScrollFreeze(freeze_cols, freeze_rows);

        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("Nazwa obrazu");
        ImGui::TableSetupColumn("Wybierz");

        ImGui::TableHeadersRow();
        int row = 0;
        for (const auto &files_iter : std::filesystem::directory_iterator(loading_image_path_))
        {
            if (!is_regular_file(files_iter) || files_iter.path().extension() != ".dat")
                continue;

            ImGui::TableNextRow();

            if (!ImGui::TableSetColumnIndex(0))
                continue;
            ImGui::Text("%d", row);

            if (!ImGui::TableSetColumnIndex(1))
                continue;

            const auto& filepath = files_iter.path();
            const std::string file_name = filepath.filename().string();

            ImGui::Text(file_name.c_str());

            if (!ImGui::TableSetColumnIndex(2))
                continue;

            bool is_enabled = selected_files_[filepath];

            ImGui::PushID(row);
            if (ImGui::Checkbox("", &is_enabled))
            {
                selected_files_[filepath] = is_enabled;
            }
            ImGui::PopID();
            row++;
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();

    if (ImGui::Button("Wczytaj"))
    {
        LoadImages();
        ImGui::CloseCurrentPopup();
    }
}

void DataInputImageWindow::LoadImages()
{
    for (const auto &[filepath, is_enabled] : selected_files_)
    {
        if (!is_enabled)
            continue;

        const FilesystemPaths paths{.envi_header = envi_header_path_, .img_data = filepath};
        const auto entity = CreateImage(paths);
        LOG_INFO("Created entity with file path: {}", filepath.string());

        selected_entity_.push_back(entity);
    }
}

CpuMatrix RunImageThreshold(const CpuMatrix& img, ThresholdSetting setting)
{
    const auto pixels_width = img.size.width * img.size.height;
    const auto data = img.data.get() + pixels_width * (setting.band - 1);

    Matrix m_img{.bands_height = 1, .pixels_width = pixels_width, .data = data};
    auto mask = ManualThresholding(m_img, 0, setting.threshold);

    mask.size.height = img.size.height;
    mask.size.width = img.size.width;
    mask.size.depth = 1;

    return std::move(mask);
}

void ThresholdPopupWindow::Show()
{
    if (ImGui::SliderInt("Pasmo##PCA_PASMO",  &selected_band_, 1, img_size_.depth, "%d", ImGuiSliderFlags_ClampOnInput))
    {
        original_img_.SetBand(selected_band_);
        RunThreshold();
    }

    if (ImGui::InputFloat(reinterpret_cast<const char *>(u8"Próg"), &threshold_value_, 0.01f, 1.f))
    {
        RunThreshold();
    }

    original_img_.Show(0.5f, 0.5f);
    ImGui::SameLine(0, 25);
    threshold_img_.Show(0.5f, 0.5f);

    if (ImGui::Button("Zapisz"))
    {
        saved_settings_ = {.threshold = threshold_value_, .band = selected_band_};
        LOG_INFO("Threshold popup window, saved settings: threshold={}, band={}", threshold_value_, selected_band_);
        ImGui::CloseCurrentPopup();
    }
}

void ThresholdPopupWindow::Load(const CpuMatrix &cpu_matrix)
{
    img_size_ = cpu_matrix.size;
    original_img_.LoadImage(cpu_matrix);
    selected_band_ = 1;
    threshold_value_ = 0.0f;

    RunThreshold();
}


void PcaPopupWindow::Show()
{
    // Run with all pca bands, then select to run all objects

    ImGui::SliderInt("Liczba pasm##PCA_PASMO",  &selected_bands_, 1, max_bands_, "%d", ImGuiSliderFlags_ClampOnInput);

    if (ImGui::Button("Zapisz"))
    {
        saved_settings_ = {.selected_bands = static_cast<std::size_t>(selected_bands_)};
        LOG_INFO("PCA popup window, saved settings: k_bands={}", selected_bands_);
        ImGui::CloseCurrentPopup();
    }
}

void StatisticWindow::Show()
{
    ImGui::SeparatorText("Parametry statystyczne");
   constexpr static ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                               ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                               ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

   constexpr static ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_SpanAllColumns;

    static int freeze_cols = 1;
    static int freeze_rows = 1;

    ImVec2 outer_size = ImVec2(0.0f, 16 * 8 * 2);
    if (ImGui::BeginTable(reinterpret_cast<const char*>(u8"Parametry statyczne dla obrazów"), 6, flags, outer_size))
    {
        ImGui::TableSetupScrollFreeze(freeze_cols, freeze_rows);

        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("Nazwa obrazu");
        ImGui::TableSetupColumn(reinterpret_cast<const char*>(u8"Średnia"));
        ImGui::TableSetupColumn(reinterpret_cast<const char*>(u8"Wariancja"));
        ImGui::TableSetupColumn(reinterpret_cast<const char*>(u8"Skośność"));
        ImGui::TableSetupColumn(reinterpret_cast<const char*>(u8"Kurtoza"));

        ImGui::TableHeadersRow();
        int row = 1;
        for (const auto &[entity, statistic_vec]: statistics_)
        {
            ImGui::TableNextRow();

            if (!ImGui::TableSetColumnIndex(0))
                continue;
            ImGui::Text("%d", row);

            ImGui::TableNextColumn();

            const auto filename = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();
            bool open = ImGui::TreeNodeEx(filename.c_str(), tree_node_flags);

            if (!open)
            {
                ++row;
                continue;
            }
            for (std::size_t pc = 0; pc < statistic_vec.size(); ++pc)
            {
                const auto idx = statistic_vec.size() - pc - 1;

                ImGui::TableNextRow();

                if (!ImGui::TableSetColumnIndex(1))
                    continue;

                std::string band_text = std::to_string(pc + 1);
                ImGui::TreeNodeEx(band_text.c_str(), tree_node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen);

                if (!ImGui::TableSetColumnIndex(2))
                    continue;
                ImGui::Text("%.2e", statistic_vec[idx].mean);

                ImGui::TableNextColumn();
                ImGui::Text("%.2e", statistic_vec[idx].variance);

                ImGui::TableNextColumn();
                ImGui::Text("%.2e", statistic_vec[idx].skewness);

                ImGui::TableNextColumn();
                ImGui::Text("%.2e", statistic_vec[idx].kurtosis);
            }

            ImGui::TreePop();
            ++row;
        }
        ImGui::EndTable();
    }
}

void StatisticWindow::Load(Entity entity, std::vector<StatisticalParameters> param)
{
    statistics_[entity] = std::move(param);
}

void DataClassificationWindow::Show()
{
    ImGui::SliderInt("Liczba klas", &class_count_, 1, 9, "%d", ImGuiSliderFlags_AlwaysClamp);


    constexpr static ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
    static int freeze_cols = 1;
    static int freeze_rows = 1;

    ImVec2 outer_size = ImVec2(0.0f, 16 * 8);
    if (ImGui::BeginTable(reinterpret_cast<const char*>(u8"Przypisz klasy"), 3, flags, outer_size))
    {
        ImGui::TableSetupScrollFreeze(freeze_cols, freeze_rows);

        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("Nazwa obrazu");
        ImGui::TableSetupColumn("Klasa");

        ImGui::TableHeadersRow();
        int row = 0;
        for (const auto entity : entities_)
        {
            const std::string file_name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();

            ImGui::TableNextRow();

            if (!ImGui::TableSetColumnIndex(0))
                continue;
            ImGui::Text("%d", row);

            if (!ImGui::TableSetColumnIndex(1))
                continue;

            ImGui::Text(file_name.c_str());

            if (!ImGui::TableSetColumnIndex(2))
                continue;

            ImGui::PushID(row);

            ImGui::SliderInt("Liczba klas", &classes_[entity], 0, class_count_, "%d", ImGuiSliderFlags_AlwaysClamp);

            ImGui::PopID();
            row++;
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();

    if (ImGui::Button("Zapisz"))
    {
        ImGui::CloseCurrentPopup();
    }
}

void DataClassificationWindow::Load(std::vector<Entity> entities)
{
    entities_ = std::move(entities);
}


void MainWindow::Show()
{
    ImGui::Begin("Ustawienia");

    if (ImGui::Button("Wczytaj dane"))
    {
        ImGui::OpenPopup("Wczytaywanie danych");
    }

    ImGui::Button("Ustawienia");
    if (ImGui::BeginCombo(reinterpret_cast<const char*>(u8"Wyświelt obraz"), selected_img_name_.c_str()))
    {
        for (const auto entity : data_input_window_.GetLoadedEntities())
        {
            ImGui::PushID(reinterpret_cast<void*>(entity));
            auto name = coordinator.GetComponent<FilesystemPaths>(entity).img_data.filename().string();

            if (ImGui::Selectable(name.c_str(), false))
            {
                selected_img_name_= name;
                threshold_window_.LoadEntity(entity);

                UpdateThresholdImage();
                UpdatePcaImage();
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    if (ImGui::Button("Progowanie"))
    {
        auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
        threshold_popup_window_.Load(cpu_img);
        ImGui::OpenPopup("Progowanie##Okno progowania");
    }

    ImGui::SameLine();
    if (ImGui::Button("PCA"))
    {
        auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
        pca_popup_window_.SetMaxBands(cpu_img.size.depth);
        ImGui::OpenPopup("Ustawienia PCA");
    }

    ImGui::SameLine();
    if (ImGui::Button("Klasyfikacja"))
    {
        data_classification_window_.Load(data_input_window_.GetLoadedEntities());
        ImGui::OpenPopup("Klasyfikacja##Klasyfikacja_okno");
    }

    ImGui::SameLine();
    if (ImGui::Button("Wszystko TEST"))
    {
        RunAllButton();
    }

    ShowPopupsWindow();

    ImGui::End();

    ImGui::Begin("Wyświetlanie");

    threshold_window_.Show();

    if (has_run_pca_)
    {
        pca_transformed_window_.Show();

        statistic_window_.Show();
    }

    ImGui::End();
}

void MainWindow::RunAllButton()
{
    if (selected_img_name_.empty())
    {
        // TODO: print error message
        LOG_WARN("RunAllButton: image is empty");
        return;
    }

    const auto start = std::chrono::high_resolution_clock::now();

    const auto &entities_vec = data_input_window_.GetLoadedEntities();
    const auto opt_threshold_settings = threshold_popup_window_.GetThresholdSettings();

    if (!opt_threshold_settings.has_value())
    {
        LOG_WARN("RunAllButton: Threshold settings not set");
        return;
    }
    const auto threshold_setting = opt_threshold_settings.value();

    const auto opt_pca_settings = pca_popup_window_.GetPcaSettings();
    if (!opt_pca_settings.has_value())
    {
        LOG_WARN("RunAllButton: PCA settings not set");
        return;
    }
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;

    ImageSize max_obj_size; // TODO make it better

    /// IMAGE PREPROCESSING
    std::vector<CpuMatrix> cpu_img_objects;
    cpu_img_objects.reserve(entities_vec.size());


    for (const auto entity : entities_vec)
    {
        const auto cpu_img = GetImageData(entity);

        max_obj_size = cpu_img.size;

        const auto mask = RunImageThreshold(cpu_img, threshold_setting);

        /// Object on mask
        auto cpu_object = GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());
        cpu_img_objects.push_back(cpu_object);
    }

    /// PCA
    auto LoadData = [&](std::size_t i) -> CpuMatrix { assert(i < cpu_img_objects.size()); return cpu_img_objects[i]; };
    result_pca_ = PCA(LoadData,  max_obj_size.depth, max_obj_size.height * max_obj_size.width, cpu_img_objects.size());

    /// Get most important eigenvectors
    result_pca_.eigenvectors = GetImportantEigenvectors(result_pca_.eigenvectors, k_bands);
    std::ostringstream oss;

    for (std::size_t i = 0; i < result_pca_.eigenvalues.size.height; ++i)
    {
        oss << result_pca_.eigenvalues.data[i] << ", ";
    }
    LOG_INFO("PCA Result: {} eigenvalues: {}", result_pca_.eigenvalues.size.height, oss.str());
    has_run_pca_ = true;
    UpdatePcaImage();


    /// TRANSFORMING IMAGE TO PCA RESULT
    const auto pca_transformed_objects = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadData,
        max_obj_size.height * max_obj_size.width, cpu_img_objects.size());

    for (std::size_t i = 0; i < entities_vec.size(); ++i)
    {
        const auto &pca_object = pca_transformed_objects[i];
        const auto entity = entities_vec[i];

        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);

        statistic_window_.Load(entity, statistic_vector);
        statistical_params_.push_back(statistic_vector);
    }

    /// CLASSFIAITON

    // TODO:
    // 2. Calculate statistical values
    // 3. Tree/SVM

    ObjectList objects;
    objects.reserve(statistical_params_.size());

    for (const auto &statistic : statistical_params_)
    {
        std::vector<float> statistic_vector;
        statistic_vector.reserve(statistical_params_.size() * 4);

        for (const auto &stat_value : statistic)
        {
            statistic_vector.push_back(stat_value.mean);
            statistic_vector.push_back(stat_value.variance);
            statistic_vector.push_back(stat_value.skewness);
            statistic_vector.push_back(stat_value.kurtosis);
        }
        objects.push_back(statistic_vector);
    }
    std::vector<uint32_t> obj_classes;
    const auto map_class = data_classification_window_.GetClasses();
    const std::size_t class_count = data_classification_window_.GetClassCount();

    for (auto entity : entities_vec)
    {
        obj_classes.push_back(map_class.at(entity));
    }

    Tree tree{};
    tree.Train(objects, obj_classes, class_count);
    tree.Print();

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("RunAll took {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

void MainWindow::ShowPopupsWindow()
{
    if (ImGui::BeginPopup("Wczytaywanie danych"))
    {
        data_input_window_.Show();
        has_run_pca_ = false;
        ImGui::EndPopup();
    }


    if (ImGui::BeginPopup("Progowanie##Okno progowania"))
    {
        threshold_popup_window_.Show();
        UpdateThresholdImage();
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Ustawienia PCA"))
    {
        pca_popup_window_.Show();
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Klasyfikacja##Klasyfikacja_okno"))
    {
        data_classification_window_.Show();
        ImGui::EndPopup();
    }
}

void MainWindow::UpdateThresholdImage()
{
    const auto opt_settings = threshold_popup_window_.GetThresholdSettings();

    if (!opt_settings.has_value())
    {
        return;
    }

    const auto [threshold, threshold_band] = opt_settings.value();
    threshold_window_.RunThreshold(threshold, threshold_band);
}

void MainWindow::UpdatePcaImage()
{
    const auto opt_entity = threshold_window_.LoadedEntity();;
    auto opt_pca_settings = pca_popup_window_.GetPcaSettings();

    if (!opt_pca_settings.has_value() || !opt_entity.has_value() || !has_run_pca_)
    {
        return;
    }
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;
    auto cpu_img = GetImageData(opt_entity.value());
    auto LoadDataImg = [=](std::size_t i) -> CpuMatrix { return cpu_img; };

    LOG_INFO("UpdatePcaImage, loading enitty id={}, pca bands={}", opt_entity.value(), k_bands);


    pca_transformed_images_ = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadDataImg, cpu_img.size.height * cpu_img.size.width, 1);

    assert(pca_transformed_images_.size() == 1);
    pca_transformed_window_.Load(pca_transformed_images_[0]);
}
