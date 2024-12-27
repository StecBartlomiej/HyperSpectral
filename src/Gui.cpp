#include "Gui.hpp"
#include "Logger.hpp"
#include "Components.hpp"
#include "Image.hpp"
#include "Classification.hpp"

#include "imnodes.h"

#include "cereal/archives/json.hpp"

#include <cassert>
#include <map>
#include <chrono>
#include <fstream>


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
    ImNodes::CreateContext();

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

    ImGuiStyle &style = ImGui::GetStyle();
    style.GrabRounding = 4.0f;
    style.FrameRounding = 4.0f;

    return window;
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

void SettingsPopupWindow()
{
    ImGuiStyle& style = ImGui::GetStyle();

    static int style_idx = -1;
    if (ImGui::Combo("Wybierz styl", &style_idx, "Ciemny\0Jasny\0"))
    {
        switch (style_idx)
        {
            case 0:
                ImGui::StyleColorsDark();
                style.FrameBorderSize  = 0.0f;
            break;
            case 1:
                ImGui::StyleColorsLight();
                style.FrameBorderSize  = 1.0f;
            break;
            default:
                break;
        }
    }
    static constexpr float min_scale = 0.5f;
    static constexpr float max_scale = 2.0f;

    ImGuiIO &io = ImGui::GetIO();
    ImGui::DragFloat("Skala interfejsu", &io.FontGlobalScale, 0.005f, min_scale, max_scale, "%.2f", ImGuiSliderFlags_AlwaysClamp);
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

            ImGui::SetNextItemWidth(16*4);
            ImGui::SliderInt("##slider_int", &classes_[entity], 0, class_count_, "%d", ImGuiSliderFlags_AlwaysClamp);

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


void TreeViewWindow::Show(const Node *root)
{
    ImNodes::BeginNodeEditor();

    const int curr_node_id = unique_node_id_++;
    ImNodes::BeginNode(curr_node_id);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Korzeń");
    ImNodes::EndNodeTitleBar();


    ImGui::Text(GetAttributeName(root->attribute_idx));
    ImGui::Text("Próg = %f", root->threshold);

    // Right ?
    const int right_node_output = unique_attr_id_++;
    ImNodes::BeginOutputAttribute(right_node_output);
    ImNodes::EndOutputAttribute();

    ImGui::Spacing();

    // Left ?
    const int left_node_output = unique_attr_id_++;
    ImNodes::BeginOutputAttribute(left_node_output);
    ImNodes::EndOutputAttribute();

    ImNodes::EndNode();

    const auto root_pos = ImNodes::GetNodeGridSpacePos(curr_node_id);

    // CALL
    if (root->right)
    {
        const ImVec2 new_pos = {root_pos.x + dx, root_pos.y - start_dy};
        ShowNode(root->right, right_node_output, new_pos, start_dy * scale_dy);
    }
    if (root->left)
    {
        const ImVec2 new_pos = {root_pos.x + dx, root_pos.y + start_dy};
        ShowNode(root->left, left_node_output, new_pos, start_dy * scale_dy);
    }

    ImNodes::EndNodeEditor();

    unique_node_id_ = 0;
    unique_attr_id_ = 0;
    unique_link_id_ = 0;
}

void TreeViewWindow::ShowNode(const Node *root, const int input_id, const ImVec2 pos, int dy)
{
    assert(root != nullptr);

    const int curr_node_id = unique_node_id_++;
    ImNodes::BeginNode(curr_node_id);

    if (IsLeaf(root))
    {
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted("Liść");
        ImNodes::EndNodeTitleBar();

        ImGui::Text("Klasa %d", root->attribute_idx);
    }
    else
    {
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted("Węzeł");
        ImNodes::EndNodeTitleBar();

        ImGui::Text(GetAttributeName(root->attribute_idx));
        ImGui::Text("Próg = %f", root->threshold);
    }

    const int node_input = unique_attr_id_++;
    ImNodes::BeginInputAttribute(node_input);
    ImNodes::EndInputAttribute();

    const int right_node_output = unique_attr_id_++;
    ImNodes::BeginOutputAttribute(right_node_output);
    ImNodes::EndOutputAttribute();

    ImGui::Spacing();

    const int left_node_output = unique_attr_id_++;
    ImNodes::BeginOutputAttribute(left_node_output);
    ImNodes::EndOutputAttribute();

    ImNodes::EndNode();

    ImNodes::Link(unique_link_id_++, node_input, input_id);

    ImNodes::SetNodeGridSpacePos(curr_node_id, pos);

    if (root->right)
    {
        const ImVec2 new_pos = {pos.x + dx, pos.y - dy};
        ShowNode(root->right, right_node_output, new_pos, dy * scale_dy);
    }
    if (root->left)
    {
        const ImVec2 new_pos = {pos.x + dx, pos.y + dy};
        ShowNode(root->left, left_node_output, new_pos, dy * scale_dy);
    }
}


void SvmViewWindow::Show()
{
    constexpr static ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
    static int freeze_cols = 1;
    static int freeze_rows = 1;

    ImVec2 outer_size = ImVec2(0.0f, 16 * 8);
    if (ImGui::BeginTable(reinterpret_cast<const char*>(u8""), 2, flags, outer_size))
    {
        ImGui::TableSetupScrollFreeze(freeze_cols, freeze_rows);

        ImGui::TableSetupColumn("Wymiar", ImGuiTableColumnFlags_NoHide);
        ImGui::TableSetupColumn("a");

        ImGui::TableHeadersRow();
        int row = 0;
        for (const auto a: alpha_)
        {
            ImGui::TableNextRow();

            if (!ImGui::TableSetColumnIndex(0))
                continue;
            ImGui::Text("%d", row);

            if (!ImGui::TableSetColumnIndex(1))
                continue;

            ImGui::Text(std::to_string(a).c_str());
            row++;
        }
        ImGui::EndTable();
    }
    ImGui::Text("b = %f", b_);
}

void SvmViewWindow::Set(std::vector<float> alpha, float b)
{
    alpha_ = std::move(alpha);
    b_ = b;
}

void MainWindow::Show()
{
    ImGui::Begin("Ustawienia");

    if (ImGui::Button("Ustawienia##Ustaw_button"))
    {
        ImGui::OpenPopup("UstawieniaPopup");
    }

    ImGui::Spacing();

    if (ImGui::Button("Wczytaj dane"))
    {
        ImGui::OpenPopup("Wczytaywanie danych");
    }

    ImGui::Spacing();

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

    ImGui::Spacing();

    static constexpr ImVec2 button_size{80, 25};

    if (ImGui::Button("Progowanie", button_size))
    {
        auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
        threshold_popup_window_.Load(cpu_img);
        ImGui::OpenPopup("Progowanie##Okno progowania");
    }

    ImGui::SameLine();
    if (ImGui::Button("PCA", button_size))
    {
        auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
        pca_popup_window_.SetMaxBands(cpu_img.size.depth);
        ImGui::OpenPopup("Ustawienia PCA");
    }

    ImGui::SameLine();
    if (ImGui::Button("Klasyfikacja", button_size))
    {
        data_classification_window_.Load(data_input_window_.GetLoadedEntities());
        ImGui::OpenPopup("Klasyfikacja##Klasyfikacja_okno");
    }

    ImGui::Spacing();

    ImGui::Checkbox(reinterpret_cast<const char*>(u8"Dodaj sąsiednie kanały"), &add_neighbour_bands_);

    ImGui::Spacing();

    ImGui::SliderInt(reinterpret_cast<const char*>(u8"K-krzyżowa walidacja"), &k_folds_,
        1, 15, "%d", ImGuiSliderFlags_AlwaysClamp);

    ImGui::Spacing();

    if (ImGui::BeginCombo(reinterpret_cast<const char*>(u8"Model uczenia"), selected_model_.data()))
    {
        constexpr static std::array<std::string_view, 2> model_names = {"Drzewo decyzyjne", "SVM"};

        int i = 0;
        for (const auto name: model_names)
        {
            ImGui::PushID(i);
            if (ImGui::Selectable(name.data(), false))
            {
                selected_model_ = name;
            }
            ImGui::PopID();
            ++i;
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();

    if (ImGui::Button("Uruchom uczenie"))
    {
        RunTrain();
    }

    ImGui::Spacing();
    if (ImGui::Button("Zapisz wyniki przetwarzania"))
    {
        SaveStatisticValues();
    }

    ShowPopupsWindow();

    ImGui::End();

    ImGui::Begin("Wyświetlanie");

    threshold_window_.Show();

    if (has_run_pca_)
    {
        pca_transformed_window_.Show();

        statistic_window_.Show();

        ImGui::SeparatorText("Wizualizacja model");

        ImGui::BeginChild("Modele uczenia", ImVec2(ImGui::GetContentRegionAvail().x, 600));
        if (ImGui::BeginTabBar("Modele uczenia##uczenie tab"))
        {
            if (ImGui::BeginTabItem("Drzewo decyzyjne"))
            {
                if (tree_.GetRoot() != nullptr)
                    tree_view_window_.Show(tree_.GetRoot());
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("SVM"))
            {
                svm_view_window_.Show();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();
    }

    ImGui::End();
}

void MainWindow::SaveStatisticValues()
{
    std::ofstream file{"Saved_data.json"};
    cereal::JSONOutputArchive archive(file);

    statistic_window_.serialize(archive);
}

void MainWindow::RunTrain()
{
    if (selected_model_.empty())
    {
        LOG_WARN("Select model before running classfication!");
        return;
    }

    if (selected_img_name_.empty())
    {
        LOG_WARN("RunTrain: image is empty");
        return;
    }

    const auto start = std::chrono::high_resolution_clock::now();

    const auto &entities_vec = data_input_window_.GetLoadedEntities();
    const auto opt_threshold_settings = threshold_popup_window_.GetThresholdSettings();

    if (entities_vec.empty())
    {
        LOG_WARN("RunTrain: empty entities vector");
        return;
    }

    if (!opt_threshold_settings.has_value())
    {
        LOG_WARN("RunTrain: Threshold settings not set");
        return;
    }
    const auto threshold_setting = opt_threshold_settings.value();

    const auto opt_pca_settings = pca_popup_window_.GetPcaSettings();
    if (!opt_pca_settings.has_value())
    {
        LOG_WARN("RunTrain: PCA settings not set");
        return;
    }
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;


    /// IMAGE PREPROCESSING
    std::vector<CpuMatrix> cpu_img_objects;
    cpu_img_objects.reserve(entities_vec.size());

    if (add_neighbour_bands_)
    {
        LOG_INFO("Running add_neighbour_bands_");
        for (const auto entity : entities_vec)
        {
            auto orginal_img = GetImageData(entity);
            auto cpu_img = AddNeighboursBand(orginal_img.GetMatrix(), orginal_img.size);
            img_size_ = cpu_img.size;

            const auto mask = RunImageThreshold(cpu_img, threshold_setting);

            /// Object on mask
            auto cpu_object = GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());
            cpu_img_objects.push_back(cpu_object);
        }
    }
    else
    {
        for (const auto entity : entities_vec)
        {
            auto cpu_img = GetImageData(entity);
            img_size_ = cpu_img.size;

            const auto mask = RunImageThreshold(cpu_img, threshold_setting);

            /// Object on mask
            auto cpu_object = GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());
            cpu_img_objects.push_back(cpu_object);
        }
    }

    ImageSize max_obj_size = img_size_;

    /// PCA
    auto LoadData = [&](std::size_t i) -> CpuMatrix { assert(i < cpu_img_objects.size()); return cpu_img_objects[i]; };
    result_pca_ = PCA(LoadData,  max_obj_size.depth, max_obj_size.height * max_obj_size.width, cpu_img_objects.size());

    /// Get most important eigenvectors
    result_pca_.eigenvectors = GetImportantEigenvectors(result_pca_.eigenvectors, k_bands);
    std::ostringstream oss;

    const auto max_i = result_pca_.eigenvalues.size.height;
    for (std::size_t i = 0; i < k_bands; ++i)
    {
        oss << result_pca_.eigenvalues.data[max_i - i - 1] << ", ";
    }
    LOG_INFO("PCA Result: {} highest eigenvalues: {}", k_bands, oss.str());
    has_run_pca_ = true;
    UpdatePcaImage();


    const auto pca_transformed_objects = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadData,
        max_obj_size.height * max_obj_size.width, cpu_img_objects.size());

    statistical_params_.clear();
    for (std::size_t i = 0; i < entities_vec.size(); ++i)
    {
        const auto &pca_object = pca_transformed_objects[i];
        const auto entity = entities_vec[i];

        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);

        statistic_window_.Load(entity, statistic_vector);
        statistical_params_.push_back(statistic_vector);
    }

    /// CLASSFIAITON, choose SVM or TREE

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

    if (selected_model_ == "Drzewo decyzyjne")
    {
        RunDecisionTree(objects, obj_classes, class_count);
    }
    else if (selected_model_ == "SVM")
    {
        LOG_INFO("Running SVM");

        auto lambda = [](const AttributeList &a, const AttributeList &b) -> float { return KernelRbf(a, b, 0.001f); };
        svm_.Train(objects, obj_classes, lambda);

        svm_view_window_.Set(svm_.GetAlpha(), svm_.GetB());
    }
    else
    {
        LOG_ERROR("Unknown model selected");
    }

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("RunAll took {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    has_run_model_ = true;
}

void MainWindow::ShowPopupsWindow()
{
   if (ImGui::BeginPopup("UstawieniaPopup"))
   {
       SettingsPopupWindow();
       ImGui::EndPopup();
   }

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

    if (add_neighbour_bands_)
    {
        cpu_img = AddNeighboursBand(cpu_img.GetMatrix(), cpu_img.size);
    }
    auto LoadDataImg = [=](std::size_t i) -> CpuMatrix { return cpu_img; };

    LOG_INFO("UpdatePcaImage, loading enitty id={}, pca bands={}", opt_entity.value(), k_bands);


    pca_transformed_images_ = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadDataImg, cpu_img.size.height * cpu_img.size.width, 1);

    assert(pca_transformed_images_.size() == 1);
    pca_transformed_window_.Load(pca_transformed_images_[0]);
}

void MainWindow::ImagePreprocessing()
{
    const auto &entities_vec = data_input_window_.GetLoadedEntities();
    const auto threshold_setting = threshold_popup_window_.GetThresholdSettings().value();

    std::vector<CpuMatrix> cpu_img_objects;
    cpu_img_objects.reserve(entities_vec.size());

    for (const auto entity : entities_vec)
    {
        const auto cpu_img = GetImageData(entity);

        img_size_ = cpu_img.size;

        const auto mask = RunImageThreshold(cpu_img, threshold_setting);

        /// Object on mask
        auto cpu_object = GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());
        cpu_img_objects.push_back(cpu_object);
    }
}

void MainWindow::RunDecisionTree(const ObjectList &objects, std::vector<uint32_t> &obj_classes, uint32_t class_count)
{
    LOG_INFO("Running decision tree");
    std::size_t max_error = objects.size();

    if (k_folds_ == 1)
    {
        LOG_INFO("Running with random data training-test 70-30 split");
        const auto [training_data, training_classes, test_data, test_classes] = SplitData(objects, obj_classes, class_count, 0.7);
        tree_.Train(training_data, training_classes, class_count);

        const auto class_result = tree_.Classify(test_data);
        const auto error = std::count_if(test_classes.cbegin(), test_classes.cend(),
                        [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });
        max_error = error;
    }
    else
    {
        LOG_INFO("Running with {}-fold cross validation", k_folds_);
        const auto folds_idx = KFoldGeneration(obj_classes, class_count, k_folds_);

        Tree tree;

        ObjectList best_training_data{};
        std::vector<uint32_t> best_training_classes;

        for (std::size_t k = 0; k < folds_idx.size(); ++k)
        {
            const auto [training_data, training_classes, test_data, test_classes] =
                GetFold(folds_idx, objects, obj_classes, k);

            tree.Train(training_data, training_classes, class_count);

            const auto class_result = tree.Classify(test_data);

            assert(class_result.size() == test_classes.size());
            const auto error = std::count_if(test_classes.cbegin(), test_classes.cend(),
                            [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });

            if (error < max_error)
            {
                max_error = error;
                best_training_data = training_data;
                best_training_classes = training_classes;
            }
        }
        tree_.Train(best_training_data, best_training_classes, class_count);

    }
    LOG_INFO("Classification error for best tree: {} ", max_error);
}

void MainWindow::RunSVM(const ObjectList &objects, std::vector<uint32_t> &obj_classes, uint32_t class_count)
{
    LOG_INFO("Running SVM");
    std::size_t max_error = objects.size();

    static constexpr float gamma = 0.001;
    auto kernel_func = [](const AttributeList &a1, const AttributeList &a2) -> float { return KernelRbf(a1, a2, gamma); };

    if (k_folds_ == 1)
    {
        LOG_INFO("Running with random data training-test 70-30 split");
        const auto [training_data, training_classes, test_data, test_classes] = SplitData(objects, obj_classes, class_count, 0.7);
        svm_.Train(training_data, training_classes, kernel_func);

        const auto class_result = svm_.Classify(test_data);
        const auto error = std::count_if(test_classes.cbegin(), test_classes.cend(),
                        [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });
        max_error = error;
    }
    else
    {
        LOG_INFO("Running with {}-fold cross validation", k_folds_);
        const auto folds_idx = KFoldGeneration(obj_classes, class_count, k_folds_);

        SVM svm;

        ObjectList best_training_data{};
        std::vector<uint32_t> best_training_classes;

        for (std::size_t k = 0; k < folds_idx.size(); ++k)
        {
            const auto [training_data, training_classes, test_data, test_classes] =
                GetFold(folds_idx, objects, obj_classes, k);

            svm.Train(training_data, training_classes, kernel_func);

            const auto class_result = svm.Classify(test_data);

            assert(class_result.size() == test_classes.size());
            const auto error = std::count_if(test_classes.cbegin(), test_classes.cend(),
                            [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });

            if (error < max_error)
            {
                max_error = error;
                best_training_data = training_data;
                best_training_classes = training_classes;
            }
        }
        svm_.Train(best_training_data, best_training_classes, kernel_func);

    }
    LOG_INFO("Classification error for best svm: {} ", max_error);
}

const char* GetAttributeName(std::size_t idx)
{
    const static std::array<const char*, 4> attr_names_ = {
        reinterpret_cast<const char*>(u8"Średnia"),
        reinterpret_cast<const char*>(u8"Wariancja"),
        reinterpret_cast<const char*>(u8"Skośność"),
        reinterpret_cast<const char*>(u8"Kurtoza")
    };

    assert(idx < attr_names_.size());
    return attr_names_[idx];
}
