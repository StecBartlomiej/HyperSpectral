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
#include <execution>
#include <fstream>
#include <numeric>
#include <ranges>
#include <random>
#include <utility>
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
    // glfwSwapInterval(1); // Enable vsync

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
        if (envi_header_path_.empty())
        {
            LOG_WARN("Select ENVI header file");
            return;
        }
        LoadImages();
        ImGui::CloseCurrentPopup();
    }
}

void DataInputImageWindow::LoadImages()
{
    selected_entity_.clear();
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
                ImNodes::StyleColorsDark();
                style.FrameBorderSize  = 0.0f;
            break;
            case 1:
                ImGui::StyleColorsLight();
                ImNodes::StyleColorsLight();
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


void ImagePatchView::Show()
{
    if (ImGui::SliderInt("Indeks##idx_patch",  &patch_index_, 1, patch_count, "%d", ImGuiSliderFlags_ClampOnInput))
    {
        patch_image_.LoadImage(patch_system_.GetPatchImage(patch_index_ - 1));
    }

    if (ImGui::SliderInt("Pasmo##PASMO_idx_patch",  &selected_band_, 1, patch_size_.depth, "%d", ImGuiSliderFlags_ClampOnInput))
    {
        patch_image_.SetBand(selected_band_);
        // RunThreshold();
    }

    // if (ImGui::InputFloat(reinterpret_cast<const char *>(u8"Próg"), &threshold_value_, 0.01f, 1.f))
    // {
    //     RunThreshold();
    // }

    patch_image_.Show(10, 10);

    // if (ImGui::Button("Zapisz"))
    // {
    //     // saved_settings_ = {.threshold = threshold_value_, .band = selected_band_};
    //     // LOG_INFO("Threshold popup window, saved settings: threshold={}, band={}", threshold_value_, selected_band_);
    //     // ImGui::CloseCurrentPopup();
    // }
}

void ImagePatchView::Load(Entity img)
{
    patch_system_.parent_img = img;
    parent_ = img;

    const auto size = coordinator.GetComponent<ImageSize>(parent_);

    patch_image_.LoadImage(patch_system_.GetPatchImage(0));

    patch_count = patch_system_.GetPatchNumbers(size);
    LOG_INFO("Patch count: {}", patch_count);

    patch_size_.width = PatchData::S;
    patch_size_.height = PatchData::S;
    patch_size_.depth = size.depth;
}

void LabelPopupWindow::Show()
{
    ImGui::SliderInt("Liczba klas",  &class_count_, 1, 16, "%d", ImGuiSliderFlags_ClampOnInput);

    static const std::filesystem::path loading_image_path_{R"(E:\Praca inzynierska\HSI images\)"};

    if (ImGui::BeginCombo(reinterpret_cast<const char*>(u8"Plki tabeli prawd"), label_file_.string().c_str()))
    {
        std::size_t idx = 0;
        for (const auto& filepath: std::filesystem::directory_iterator(loading_image_path_))
        {
            if (!filepath.is_regular_file() || filepath.path().extension() != ".dat")
                continue;

            ImGui::PushID(reinterpret_cast<void*>(idx));
            const auto name = filepath.path().filename().string();

            bool selected = false;
            if (ImGui::Selectable(name.c_str(), selected))
            {
                label_file_ = filepath.path();
            }
            ImGui::PopID();
            ++idx;
        }
        ImGui::EndCombo();
    }

    if (ImGui::Button("Zapisz"))
    {
        ImGui::CloseCurrentPopup();
    }
}

void TreeViewWindow::Show(const Node *root)
{
    ImGui::PushItemWidth(80);
    ImGui::InputInt("dx", &dx);
    ImGui::SameLine();
    ImGui::InputInt("start dy", &start_dy);

    ImGui::InputInt("leaf dx", &leaf_dx);
    ImGui::SameLine();
    ImGui::InputInt("leaf dy", &leaf_dy);

    ImGui::DragFloat("y scale", &scale_dy, 0.01, 0, 1);
    ImGui::SameLine();
    ImGui::DragFloat("start y_scale", &start_scale_dy, 0.01, 0, 3);
    ImGui::PopItemWidth();

    ImNodes::BeginNodeEditor();

    const int curr_node_id = unique_node_id_++;
    ImNodes::BeginNode(curr_node_id);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted("Korzeń");
    ImNodes::EndNodeTitleBar();


    ImGui::Text("PC %llu", root->attribute_idx / 4u);
    ImGui::Text(GetAttributeName(root->attribute_idx % 4));
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
        const ImVec2 new_pos = {root_pos.x + dx, root_pos.y - start_dy * start_scale_dy};
        ShowNode(root->right, right_node_output, new_pos, start_dy * scale_dy);
    }
    if (root->left)
    {
        const ImVec2 new_pos = {root_pos.x + dx, root_pos.y + start_dy * start_scale_dy};
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

        ImGui::Text("PC %llu", root->attribute_idx / 4u);
        ImGui::Text(GetAttributeName(root->attribute_idx % 4));
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
        const ImVec2 new_pos = IsLeaf(root->right) ? ImVec2{pos.x + leaf_dx, pos.y - leaf_dy}: ImVec2{pos.x + dx, pos.y - dy};

        ShowNode(root->right, right_node_output, new_pos, dy * scale_dy);
    }
    if (root->left)
    {
        const ImVec2 new_pos = IsLeaf(root->left) ? ImVec2{pos.x + leaf_dx, pos.y + leaf_dy}: ImVec2{pos.x + dx, pos.y + dy};
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
    constexpr static std::array<std::string_view, 2> approach_names = {"Obiekt", "Piksel"};
    ImGui::SetNextItemWidth(100.f);
    if (ImGui::BeginCombo(reinterpret_cast<const char*>(u8"Sposób przetwarzania"), approach_names[approach_type_].data()))
    {
        int i = 0;
        for (const auto name: approach_names)
        {
            ImGui::PushID(i);
            if (ImGui::Selectable(name.data(), false))
            {
                approach_type_ = i;
            }
            ImGui::PopID();
            ++i;
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Uczenie");

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

    if (approach_type_ == 0)
    {
        ShowObjectApproach();
    }
    else
    {
        ShowPixelApproach();
    }

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

    if (selected_model_ == "SVM")
    {
        ImGui::PushItemWidth(120);

        ImGui::DragFloat("C", &params_svm_.C, 1, 1, 1e6, "%.2f");
        ImGui::InputFloat("tau", &params_svm_.tau, 0, 0, "%.5f");
        ImGui::InputFloat("Gamma", &params_svm_.gamma, 0, 0, "%.5f");
        static int max_iter = 1e5;
        if (ImGui::InputInt("Maks iteracji", &max_iter))
        {
            params_svm_.max_iter = static_cast<std::size_t>(max_iter);
        }

        ImGui::PopItemWidth();
    }

    ImGui::Spacing();
    if (ImGui::Button("Uruchom uczenie"))
    {
        RunModels();
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Testowanie");

    if (has_run_model_)
    {
        if (ImGui::Button("Wczytaj dane testowe"))
        {
            ImGui::OpenPopup("Wczytaywanie danych testowych");
        }
        ImGui::Spacing();

        if (ImGui::Button("Zapisz wyniki"))
        {
            SaveTestClassification();
        }
    }
    else
    {
        ImGui::Text("Brak modelu!");
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

        ImGui::BeginChild("Modele uczenia", ImVec2(ImGui::GetContentRegionAvail().x, 900));
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

void MainWindow::SaveTestClassification()
{
    std::ofstream file{"test_data_classification.json"};

    const auto test_entities = test_input_window_.GetLoadedEntities();
    const auto result = RunClassify(test_entities);
    SaveClassificationResult(test_entities, result, file);
}

void MainWindow::RunModels()
{
    const std::vector<Entity> &entities_vec = data_input_window_.GetLoadedEntities();

    // Split data
    if (approach_type_ == 1)
    {
        LOG_INFO("Running training with disjoint sampling");
        if (entities_vec.size() != 1)
        {
            LOG_WARN("Load only one image to run disjoint sampling!");
            return;
        }
        const auto curr_entity = entities_vec.front();

        RunTrainDisjoint(curr_entity);
    }
    else if (k_folds_ == 1)
    {
        const std::vector<uint32_t> obj_classes = GetObjectClasses(entities_vec);
        const std::size_t class_count = data_classification_window_.GetClassCount();

        LOG_INFO("Running with random data training-test 70-30 split");
        const auto [training_entity, training_classes, test_entity, test_classes] = SplitData(entities_vec, obj_classes, class_count, 0.7);

        LOG_INFO("Training data id: {}", fmt::join(training_entity, ", "));
        RunTrain(training_entity);

        LOG_INFO("Classify data id: {}", fmt::join(test_entity, ", "));
        const auto class_result = RunClassify(test_entity);

        assert(class_result.size() == test_classes.size());
        const auto error = std::count_if(test_classes.cbegin(), test_classes.cend(),
                        [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });

        std::ofstream file("Wynik_klasyfikacji_dla_zbioru_walidacyjnego.json");
        SaveClassificationResult(test_entity, class_result, file);
        LOG_INFO("Classification score for test values: {}. Result saved to file \"Wynik_klasyfikacji_dla_zbioru_walidacyjnego.json\"", error);
    }
    else if (k_folds_ >= 2)
    {
        const std::vector<uint32_t> obj_classes = GetObjectClasses(entities_vec);
        const std::size_t class_count = data_classification_window_.GetClassCount();

        LOG_INFO("Running with {}-fold cross validation", k_folds_);
        const auto folds_idx = KFoldGeneration(obj_classes, class_count, k_folds_);


        std::size_t best_error = entities_vec.size();
        std::size_t best_test_fold_idx = 0;
        for (std::size_t test_fold_idx = 0; test_fold_idx < folds_idx.size(); ++test_fold_idx)
        {
            const auto [training_entity, training_classes, test_entity, test_classes] =
                GetFold(folds_idx, entities_vec, obj_classes, test_fold_idx);

            LOG_INFO("Training data id: {}", fmt::join(training_entity, ", "));
            RunTrain(training_entity);

            LOG_INFO("Classify data id: {}", fmt::join(test_entity, ", "));
            const auto class_result = RunClassify(test_entity);

            assert(class_result.size() == test_classes.size());
            const auto error = std::count_if(test_classes.cbegin(), test_classes.cend(),
                            [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });

            if (error < best_error)
            {
                best_error = error;
                best_test_fold_idx = test_fold_idx;
            }
            LOG_INFO("Classification score for idx {} test fold: {}", test_fold_idx, error);
        }
        const auto [training_entity, training_classes, test_entity, test_classes] =
            GetFold(folds_idx, entities_vec, obj_classes, best_test_fold_idx);

        RunTrain(training_entity);
        const auto class_result = RunClassify(test_entity);

        std::ofstream file("Wynik_klasyfikacji_dla_zbioru_walidacyjnego.json");
        SaveClassificationResult(test_entity, class_result, file);
        LOG_INFO("Best classification score for test values: {}. Result saved to file \"Wynik_klasyfikacji_dla_zbioru_walidacyjnego.json\"", best_error);
    }
}

std::vector<uint32_t> MainWindow::GetObjectClasses(const std::vector<Entity> &entities)
{
    std::vector<uint32_t> obj_classes;
    const auto map_class = data_classification_window_.GetClasses();

     for (auto entity : entities)
    {
        obj_classes.push_back(map_class.at(entity));
    }
    return obj_classes;
}

void MainWindow::RunTrain(const std::vector<Entity> &entities_vec)
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
    if (entities_vec.empty())
    {
        LOG_WARN("RunTrain: empty entities vector");
        return;
    }
    const auto start = std::chrono::high_resolution_clock::now();

    const auto opt_pca_settings = pca_popup_window_.GetPcaSettings();
    if (!opt_pca_settings.has_value())
    {
        LOG_WARN("RunTrain: PCA settings not set");
        return;
    }
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;

    /// IMAGE PREPROCESSING
    std::vector<CpuMatrix> cpu_img_objects = RunThresholding(entities_vec);

    /// PCA
    RunPca(entities_vec);

    auto LoadData = [&](std::size_t i) -> CpuMatrix { assert(i < cpu_img_objects.size()); return cpu_img_objects[i]; };
    ImageSize max_obj_size = img_size_;

    /// Get statistical values
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
    const auto obj_classes = GetObjectClasses(entities_vec);
    const auto class_count = obj_classes.size();
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

    if (selected_model_ == "Drzewo decyzyjne")
    {
        LOG_INFO("Running decision tree");
        tree_.Train(objects, obj_classes, class_count);
    }
    else if (selected_model_ == "SVM")
    {
        LOG_INFO("Running SVM");

        ensemble_svm_.SetParameterSvm(class_count, params_svm_);
        ensemble_svm_.Train(objects, obj_classes);

        // svm_view_window_.Set(svm_.GetAlpha(), svm_.GetB());
    }
    else
    {
        LOG_ERROR("Unspecified model!");
        throw std::runtime_error("");
    }

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("Training took {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    has_run_model_ = true;
}

void SaveGroundTruth(const std::vector<PatchData> &patches, const std::vector<uint32_t> &class_result, ImageSize size,
    std::string_view file_name)
{
    std::ofstream file(file_name.data(), std::ios::binary);
    std::vector<uint8_t> image_data(size.width * size.height, 0u);

    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        const auto [center_x, center_y] = patches[i];
        const uint8_t curr_class = class_result[i] + 1;

        image_data[center_y * size.width + center_x] = curr_class;
    }
    for (const uint8_t x: image_data)
    {
        file << x;
    }
}

void MainWindow::RunTrainDisjoint(Entity image)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const auto size = coordinator.GetComponent<ImageSize>(image);

    const auto label_file_path = label_popup_window_.GetLabelFile();
    if (label_file_path.empty())
    {
        LOG_WARN("RunTrain: label file path not set");
        return;
    }
    ImageLabel img_label{label_file_path, size};

    const auto threshold_setting = [&]{
        const auto opt_threshold_settings = threshold_popup_window_.GetThresholdSettings();
        if (!opt_threshold_settings.has_value())
        {
            LOG_WARN("RunTrain: Threshold settings not set, running with threshold=0, band=0");
        }
        return opt_threshold_settings.value_or(ThresholdSetting{0.f, 0});
    }();

    const auto mask = [&]{
        auto original_img = GetImageData(image);
        return RunImageThreshold(original_img, threshold_setting);
    }();

    img_size_.width = PatchData::S;
    img_size_.height= PatchData::S;
    img_size_.depth = size.depth;

    LOG_INFO("Get patch positions");
    std::vector<PatchData> patch_positions;
    std::vector<uint8_t> patch_label;
    std::vector<PatchData> unknown_patch_positions;

    for (std::size_t y = 0; y < mask.size.height; ++y)
    {
        for (std::size_t x = 0; x < mask.size.width; ++x)
        {
            if (mask.data[y * mask.size.width + x] == 0.f)
                continue;

            const uint8_t label = img_label.GetLabels({x, y});
            if (label == 0)
            {
                unknown_patch_positions.emplace_back(x, y);
                continue;
            }

            patch_positions.emplace_back(x, y);
            patch_label.push_back(label - 1);
        }
    }
    assert(!patch_positions.empty());
    assert(patch_positions.size() == patch_label.size());

    {
        std::random_device rd;
        std::mt19937 g(rd());

        std::vector<std::size_t> indexes(patch_positions.size(), 0);
        std::iota(indexes.begin(), indexes.end(), 0);
        std::ranges::shuffle(indexes, g);

        std::vector<PatchData> tmp_patch_positions;
        std::vector<uint8_t> tmp_patch_labels;

        tmp_patch_positions.reserve(patch_positions.size());
        tmp_patch_labels.reserve(patch_label.size());

        std::ranges::transform(indexes, std::back_inserter(tmp_patch_positions),
                               [&](std::size_t index) {
                                   return patch_positions[index];
                               });
        std::ranges::transform(indexes, std::back_inserter(tmp_patch_labels),
                               [&](std::size_t index) {
                                   return patch_label[index];
                               });

        patch_positions = tmp_patch_positions;
        patch_label = tmp_patch_labels;
    }

    /// SPLIT DATA -> TRAIN, VALIDATION, TEST
    const auto class_count = label_popup_window_.GetClassCount();

    LOG_INFO("Splitting learning-test data {}", disjoint_data_split_);
    const auto [train_val_patch, train_val_labels, test_patch, test_labels] = SplitData(
        patch_positions, patch_label, class_count, disjoint_data_split_);

    LOG_INFO("Splitting training-validation data {}", disjoint_validation_split_);
    const auto [training_patch, training_labels, validation_patch, validation_labels] = SplitData(
        train_val_patch, train_val_labels, class_count, disjoint_validation_split_);

    LOG_INFO("Running preprocessing");
    auto [objects, obj_classes] = RunTrainPreprocessing(training_patch, training_labels, image);
    LOG_INFO("Ended preprocessing");


    LOG_INFO("Training classification");
    if (selected_model_ == "Drzewo decyzyjne")
    {
        LOG_INFO("Running decision tree");
        // Add k-fold validation

        tree_.Train(objects, obj_classes, class_count);
        // Add prunning


        // TRAINING DATA
        {
            const auto class_result = tree_.Classify(objects);
            const auto error_train = std::ranges::count_if(obj_classes,
                                                     [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });
            const auto relative_error = (static_cast<float>(error_train) / static_cast<float>(obj_classes.size())) * 100.f;
            LOG_INFO("Classification result of training data errors={}, relative={}%", error_train, relative_error);

            SaveGroundTruth(training_patch, class_result, size, "training_values.dat");
        }

        // TEST DATA
        {
            const auto test_data = RunPreprocessing(test_patch, image);

            LOG_INFO("Running classification");

            const auto class_result_2 = tree_.Classify(test_data);
            const auto error_test = std::ranges::count_if(test_labels,
                                                     [&, i=0](uint32_t c) mutable { return c != class_result_2[i++]; });
            const auto relative_error_test = (static_cast<float>(error_test) / static_cast<float>(test_labels.size())) * 100.f;
            LOG_INFO("Classification result: errors={}, relative={}%", error_test, relative_error_test);

            SaveGroundTruth(test_patch, class_result_2, size, "test_values.dat");
        }

        // Unknown data
        {
            const auto unknown_data = RunPreprocessing(unknown_patch_positions, image);
            const auto unknown_classes = tree_.Classify(unknown_data);

            SaveGroundTruth(unknown_patch_positions, unknown_classes, size, "unknown_values.dat");
        }
    }
    else if (selected_model_ == "SVM")
    {
        LOG_INFO("Running SVM");

        ensemble_svm_.SetParameterSvm(class_count, params_svm_);
        ensemble_svm_.Train(objects, obj_classes);

        {
            const auto val_data = RunPreprocessing(validation_patch, image);
            LOG_INFO("Classification of validation data");

            const auto class_result = ensemble_svm_.Classify(val_data);

            const auto error_test = std::ranges::count_if(validation_labels,
                                                     [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });
            const auto relative_error_test = (static_cast<float>(error_test) / static_cast<float>(validation_labels.size())) * 100.f;

            LOG_INFO("Classification of validation data: errors={}, relative={}%", error_test, relative_error_test);
            SaveGroundTruth(training_patch, class_result, size, "svm_validation_values.dat");
        }
        // svm_view_window_.Set(svm_.GetAlpha(), svm_.GetB());

        {
            LOG_INFO("Classification of test data");
            const auto test_data = RunPreprocessing(test_patch, image);
            LOG_INFO("Running classification");
            const auto class_result = ensemble_svm_.Classify(test_data);
            LOG_INFO("Ended classification");

            const auto error = std::ranges::count_if(test_labels,
                                                     [&, i=0](uint32_t c) mutable { return c != class_result[i++]; });

            LOG_INFO("Classification result test data: errors={}, relative={}%", error, error / static_cast<float>(test_labels.size()));
            SaveGroundTruth(test_patch, class_result, size, "svm_test_values.dat");
        }
    }
    else
    {
        LOG_ERROR("Unspecified model!");
        throw std::runtime_error("");
    }

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("Training took {} s", std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

    has_run_model_ = true;
}

ClassificationData MainWindow::RunTrainPreprocessing(const std::vector<PatchData> &patch_positions,
     const std::vector<uint8_t> &patch_label, Entity image)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const auto size = coordinator.GetComponent<ImageSize>(image);
    const ImageSize patch_size = add_neighbour_bands_ ?
        ImageSize{PatchData::S - 2, PatchData::S - 2, size.depth * 9} : ImageSize{PatchData::S, PatchData::S, size.depth};
    const auto threshold_setting = threshold_popup_window_.GetThresholdSettings().value_or(ThresholdSetting{0.f, 0});

    PatchSystem patch_system{};
    patch_system.parent_img = image;

    LOG_INFO("Run PCA on patches");
    const auto opt_pca_settings = pca_popup_window_.GetPcaSettings();
    if (!opt_pca_settings.has_value())
    {
        LOG_WARN("RunTrain: PCA settings not set");
        throw std::runtime_error("PCA settings not set");
    }
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;

    if  (add_neighbour_bands_)
        LOG_INFO("Adding neighbour bands");

    /// PCA
    auto LoadData = [&](std::size_t i) -> CpuMatrix {
        assert(i < patch_positions.size());

        const auto [x, y] = patch_positions[i];
        const auto patch_idx = y * size.width + x;

        CpuMatrix patch = patch_system.GetPatchImage(patch_idx);

        if (add_neighbour_bands_)
            patch = AddNeighboursBand(patch.GetMatrix(), patch.size);

        const auto patch_mask = RunImageThreshold(patch, threshold_setting);

        return GetObjectFromMask(patch.GetMatrix(), patch_mask.GetMatrix());
    };

    result_pca_ = PCA(LoadData,  patch_size.depth, patch_size.height * patch_size.width, patch_positions.size());
    result_pca_.eigenvectors = GetImportantEigenvectors(result_pca_.eigenvectors, k_bands);

    const auto max_i = result_pca_.eigenvalues.size.height;
    std::ostringstream oss;
    for (std::size_t i = 0; i < k_bands; ++i)
    {
        oss << result_pca_.eigenvalues.data[max_i - i - 1] << ", ";
    }
    LOG_INFO("PCA Result: {} highest eigenvalues: {}", k_bands, oss.str());
    has_run_pca_ = true;

    // UpdatePcaImage();

    LOG_INFO("Run projection of PCA on patches");
    auto patches = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadData,
        patch_size.height * patch_size.width, patch_positions.size());

    LOG_INFO("Calculate statistiacal params on patches");
    statistical_params_.clear();
    for (const auto & pca_object : patches)
    {
        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);
        statistical_params_.push_back(statistic_vector);
    }

    ObjectList objects = GetNormalizedData(statistical_params_);
    std::vector<uint32_t> obj_class;
    objects.resize(patch_label.size());
    std::ranges::copy(patch_label, std::back_inserter(obj_class));

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("Training preprocessing took {} s", std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

    return {std::move(objects), std::move(obj_class)};
}

ObjectList MainWindow::RunPreprocessing(const std::vector<PatchData> &patch_positions, Entity image)
{
    const auto start = std::chrono::high_resolution_clock::now();

    const auto size = coordinator.GetComponent<ImageSize>(image);
    const auto threshold_setting = threshold_popup_window_.GetThresholdSettings().value_or(ThresholdSetting{0.f, 0});
    const std::size_t k_bands = pca_popup_window_.GetPcaSettings().value().selected_bands;

    const ImageSize patch_size = add_neighbour_bands_ ?
        ImageSize{PatchData::S - 2, PatchData::S - 2, size.depth * 9} : ImageSize{PatchData::S, PatchData::S, size.depth};

    PatchSystem patch_system{};
    patch_system.parent_img = image;

    auto LoadData = [&](std::size_t i) -> CpuMatrix {
        assert(i < patch_positions.size());

        const auto [x, y] = patch_positions[i];
        const auto patch_idx = y * size.width + x;

        CpuMatrix patch = patch_system.GetPatchImage(patch_idx);

        if (add_neighbour_bands_)
            patch = AddNeighboursBand(patch.GetMatrix(), patch.size);

        const auto patch_mask = RunImageThreshold(patch, threshold_setting);

        return GetObjectFromMask(patch.GetMatrix(), patch_mask.GetMatrix());
    };
   LOG_INFO("Run projection of PCA on patches");
    auto patches = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadData,
        patch_size.height * patch_size.width, patch_positions.size());

    LOG_INFO("Calculate statistical params on patches");
    statistical_params_.clear();
    for (const auto & pca_object : patches)
    {
        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);
        statistical_params_.push_back(statistic_vector);
    }

    ObjectList objects;
    objects.reserve(statistical_params_.size());

    for (const auto &statistic : statistical_params_)
    {
        std::vector<float> statistic_vector;
        statistic_vector.reserve(statistical_params_.size() * 4);

        for (const auto idx : std::views::iota(0u, statistic.size()))
        {
            const auto &stat_value = statistic[idx];

            const auto &[mean, var, skew, kurt] = normalization_data_[idx];

            const float diff_mean = mean.max - mean.min;
            const float diff_var =  var.max - var.min;
            const float diff_skew = skew.max - skew.min;
            const float diff_kurt = kurt.max - kurt.min;

            const float n_mean = (stat_value.mean - mean.min) / diff_mean;
            const float n_variance = (stat_value.variance - var.min) / diff_var;
            const float n_skew = (stat_value.skewness - skew.min) / diff_skew;
            const float n_kurt = (stat_value.kurtosis - kurt.min) / diff_kurt;

            statistic_vector.push_back(n_mean);
            statistic_vector.push_back(n_variance);
            statistic_vector.push_back(n_skew);
            statistic_vector.push_back(n_kurt);
        }
        objects.push_back(statistic_vector);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("Preprocessing took {} s", std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

    return std::move(objects);
}


std::vector<uint32_t> MainWindow::RunClassify(const std::vector<Entity> &entities_vec)
{
    assert(!entities_vec.empty());
    assert(has_run_model_);

    const auto opt_pca_settings = pca_popup_window_.GetPcaSettings();
    assert(opt_pca_settings.has_value());
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;

    // DATA PREPROCESSING
    std::vector<CpuMatrix> cpu_img_objects = RunThresholding(entities_vec);

    const ImageSize max_obj_size = img_size_;

    // Transforming to PCA dimensions
    auto LoadData = [&](std::size_t i) -> CpuMatrix { assert(i < cpu_img_objects.size()); return cpu_img_objects[i]; };
    const auto pca_transformed_objects = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadData,
        max_obj_size.height * max_obj_size.width, cpu_img_objects.size());

    // Getting statistical values
    std::vector<std::vector<StatisticalParameters>> statistical_params{};
    for (std::size_t i = 0; i < entities_vec.size(); ++i)
    {
        const auto &pca_object = pca_transformed_objects[i];
        const auto entity = entities_vec[i];

        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);

        statistic_window_.Load(entity, statistic_vector);
        statistical_params.push_back(statistic_vector);
    }

    /// Classification
    ObjectList objects;
    objects.reserve(statistical_params.size());

    for (const auto &statistic : statistical_params)
    {
        std::vector<float> statistic_vector;
        statistic_vector.reserve(statistical_params.size() * 4);

        for (const auto &stat_value : statistic)
        {
            statistic_vector.push_back(stat_value.mean);
            statistic_vector.push_back(stat_value.variance);
            statistic_vector.push_back(stat_value.skewness);
            statistic_vector.push_back(stat_value.kurtosis);
        }
        objects.push_back(statistic_vector);
    }

    if (selected_model_ == "Drzewo decyzyjne")
    {
        LOG_INFO("Classification using decision tree");
        return tree_.Classify(objects);
    }
    else if (selected_model_ == "SVM")
    {
        LOG_INFO("Classification using SVM");
        return ensemble_svm_.Classify(objects);;
    }

    LOG_ERROR("Unspecified model");
    throw std::runtime_error("");
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

    if (ImGui::BeginPopup("Wczytaywanie danych testowych"))
    {
        test_input_window_.Show();
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Okno patch"))
    {
        patch_view_.Show();
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Tabela prawdy"))
    {
        label_popup_window_.Show();
        ImGui::EndPopup();
    }

}

void MainWindow::ShowPixelApproach()
{
    static constexpr ImVec2 button_size{80, 25};

    if (ImGui::Button("Progowanie", button_size))
    {
        if (selected_img_name_.empty())
        {
            LOG_WARN(reinterpret_cast<const char*>(u8"Wyświelt obraz przed progowaniem"));
        }
        else
        {
            auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
            threshold_popup_window_.Load(cpu_img);
            ImGui::OpenPopup("Progowanie##Okno progowania");
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("PCA", button_size))
    {
        auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
        pca_popup_window_.SetMaxBands(cpu_img.size.depth);
        ImGui::OpenPopup("Ustawienia PCA");
    }

    ImGui::SameLine();
    if (ImGui::Button(reinterpret_cast<const char*>(u8"Tabela prawd"), button_size))
    {
        ImGui::OpenPopup("Tabela prawdy");
    }

    ImGui::Spacing();

    ImGui::Checkbox(reinterpret_cast<const char*>(u8"Dodaj sąsiednie kanały"), &add_neighbour_bands_);

    ImGui::Spacing();

    if (ImGui::Button(reinterpret_cast<const char*>(u8"Zobacz próbki")))
    {
        auto opt_entity = threshold_window_.LoadedEntity();
        if (!opt_entity.has_value())
        {
            LOG_WARN("Load entity before viewing patches");
        }
        else
        {
            patch_view_.Load(opt_entity.value());
            ImGui::OpenPopup("Okno patch");
        }
    }
    ImGui::Spacing();


    ImGui::PushItemWidth(150);
    ImGui::DragFloat(reinterpret_cast<const char*>(u8"Podział uczące-testowe"), &disjoint_data_split_, 0.05, 0, 1);
    ImGui::Spacing();
    ImGui::DragFloat(reinterpret_cast<const char*>(u8"Podział uczące-walidacyjne"), &disjoint_validation_split_, 0.05, 0, 1);
    ImGui::Spacing();
}

void MainWindow::ShowObjectApproach()
{
  static constexpr ImVec2 button_size{80, 25};

    if (ImGui::Button("Progowanie", button_size))
    {
        if (selected_img_name_.empty())
        {
            LOG_WARN(reinterpret_cast<const char*>(u8"Wyświelt obraz przed progowaniem"));
        }
        else
        {
            auto cpu_img = GetImageData(threshold_window_.LoadedEntity().value());
            threshold_popup_window_.Load(cpu_img);
            ImGui::OpenPopup("Progowanie##Okno progowania");
        }
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
    LOG_INFO("Updating Pca image...");
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;
    auto cpu_img = GetImageData(opt_entity.value());

    if (add_neighbour_bands_)
    {
        try
        {
            cpu_img = AddNeighboursBand(cpu_img.GetMatrix(), cpu_img.size);
        }
        catch (const std::bad_alloc& err)
        {
            LOG_INFO("Showing PCA image caught out-of-memory expectation unable to process full image, {}", err.what());
            return;
        }
    }
    auto LoadDataImg = [=](std::size_t i) -> CpuMatrix { return cpu_img; };

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

void MainWindow::RunPca(const std::vector<Entity> &entities_vec)
{
    if (selected_img_name_.empty())
    {
        LOG_WARN("RunPCA: image is empty");
        return;
    }
    if (entities_vec.empty())
    {
        LOG_WARN("RunPCA: empty entities vector");
        return;
    }
    const auto start = std::chrono::high_resolution_clock::now();

    const auto opt_pca_settings = pca_popup_window_.GetPcaSettings();
    if (!opt_pca_settings.has_value())
    {
        LOG_WARN("RunTrain: PCA settings not set");
        return;
    }
    const std::size_t k_bands = opt_pca_settings.value().selected_bands;

    /// IMAGE PREPROCESSING
    std::vector<CpuMatrix> cpu_img_objects = RunThresholding(entities_vec);
    ImageSize max_obj_size = img_size_;

    /// PCA
    LOG_INFO("Starts PCA");
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
    LOG_INFO("Ended PCA");

    const auto end = std::chrono::high_resolution_clock::now();
    LOG_INFO("Training took {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
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

std::vector<CpuMatrix> MainWindow::RunThresholding(const std::vector<Entity> &entities_vec)
{
    const auto opt_threshold_settings = threshold_popup_window_.GetThresholdSettings();
    if (!opt_threshold_settings.has_value())
    {
        LOG_WARN("RunTrain: Threshold settings not set, running with threshold=0, band=0");
    }
    const auto threshold_setting = opt_threshold_settings.value_or(ThresholdSetting{0.f, 0});

    std::vector<CpuMatrix> cpu_img_objects;
    cpu_img_objects.reserve(entities_vec.size());

    auto max_img_size = coordinator.GetComponent<ImageSize>(entities_vec.front());
    if (add_neighbour_bands_)
    {
        LOG_INFO("Adding neighbour bands for texture analysis");
        max_img_size.depth *= 9;
        max_img_size.width = 0;
        max_img_size.height = 1;

        for (const auto entity : entities_vec)
        {
            CpuMatrix cpu_object = [&]() {
                auto original_img = GetImageData(entity);
                auto cpu_img = AddNeighboursBand(original_img.GetMatrix(), original_img.size);

                const auto mask = RunImageThreshold(cpu_img, threshold_setting);

                // Get size of mask
                float pixel_width = SumAllCuda(mask.GetMatrix());
                LOG_INFO("Pixel count in mask {}", pixel_width);
                if (pixel_width > max_img_size.width)
                {
                    max_img_size.width = static_cast<uint32_t>(pixel_width);
                }

                /// Object on mask
                return GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());
            }();

            cpu_img_objects.push_back(std::move(cpu_object));
        }
    }
    else
    {
        max_img_size.width = 0;
        max_img_size.height = 1;
        for (const auto entity : entities_vec)
        {
            auto cpu_img = GetImageData(entity);
            img_size_ = cpu_img.size;

            const auto mask = RunImageThreshold(cpu_img, threshold_setting);

            float pixel_width = SumAllCuda(mask.GetMatrix());
            LOG_INFO("Pixel count in mask {}", pixel_width);
            if (pixel_width > max_img_size.width)
            {
                max_img_size.width = static_cast<uint32_t>(pixel_width);
            }


            /// Object on mask
            auto cpu_object = GetObjectFromMask(cpu_img.GetMatrix(), mask.GetMatrix());
            cpu_img_objects.push_back(cpu_object);
        }
    }
    LOG_INFO("Ended thresholding");
    img_size_ = max_img_size;
    return std::move(cpu_img_objects);
}

ObjectList MainWindow::GetNormalizedData(const std::vector<std::vector<StatisticalParameters>> &statistical_params)
{
    assert(!statistical_params.empty());

    ObjectList objects;
    objects.reserve(statistical_params.size());

    // NORMALIZATION FOR EACH PC
    static constexpr float max_float = std::numeric_limits<float>::max();
    static constexpr float min_float = std::numeric_limits<float>::min();

    const std::size_t bands_size = statistical_params.front().size();

    normalization_data_.reserve(bands_size);
    for (std::size_t i = 0; i < bands_size; ++i)
    {
        NormalizationData normalization_data{
            .mean     = {.min = max_float, .max = min_float},
            .variance = {.min = max_float, .max = min_float},
            .skewness = {.min = max_float, .max = min_float},
            .kurtosis = {.min = max_float, .max = min_float},
        };
        normalization_data_.push_back(normalization_data);
    }

    for (const auto &statistic : statistical_params)
    {
        for (const auto idx : std::views::iota(0u, statistic.size()))
        {
            const auto &stat_value = statistic[idx];

            auto  &[mean, var, skew, kurt] = normalization_data_[idx];

            if (stat_value.mean > mean.max)
            {
                mean.max = stat_value.mean;
            }
            if (stat_value.mean < mean.min)
            {
                mean.min = stat_value.mean;
            }

            if (stat_value.variance > var.max)
            {
                var.max = stat_value.variance;
            }
            if (stat_value.variance < var.min)
            {
                var.min = stat_value.variance;
            }

            if (stat_value.skewness > skew.max)
            {
                skew.max = stat_value.skewness;
            }
            if (stat_value.skewness < skew.min)
            {
                skew.min = stat_value.skewness;
            }

            if (stat_value.kurtosis > kurt.max)
            {
                kurt.max = stat_value.kurtosis;
            }
            if (stat_value.kurtosis < kurt.min)
            {
                kurt.min = stat_value.kurtosis;
            }
        }
    }

    for (const auto &statistic : statistical_params)
    {
        std::vector<float> statistic_vector;
        statistic_vector.reserve(statistic.size() * 4);


        for (const auto idx : std::views::iota(0u, statistic.size()))
        {
            const auto &stat_value = statistic[idx];

            const auto &[mean, var, skew, kurt] = normalization_data_[idx];

            const float diff_mean = mean.max - mean.min;
            const float diff_var =  var.max - var.min;
            const float diff_skew = skew.max - skew.min;
            const float diff_kurt = kurt.max - kurt.min;

            const float n_mean = (stat_value.mean - mean.min) / diff_mean;
            const float n_variance = (stat_value.variance - var.min) / diff_var;
            const float n_skew = (stat_value.skewness - skew.min) / diff_skew;
            const float n_kurt = (stat_value.kurtosis - kurt.min) / diff_kurt;

            statistic_vector.push_back(n_mean);
            statistic_vector.push_back(n_variance);
            statistic_vector.push_back(n_skew);
            statistic_vector.push_back(n_kurt);
        }
        objects.push_back(statistic_vector);
    }

    return objects;
}
