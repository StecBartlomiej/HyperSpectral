#ifndef GUI_HPP
#define GUI_HPP

#include "Classification.hpp"

#include "Components.hpp"
#include "EntityComponentSystem.hpp"
#include "Image.hpp"

#include <glad/gl.h>
#include <map>
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/map.hpp"

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


class ImageViewWindow
{
public:
    void Show();

    void LoadEntity(Entity entity);

    void RunThreshold(float threshold, std::size_t threshold_band);

    [[nodiscard]] CpuMatrix GetThresholdMask() const;

    [[nodiscard]] auto LoadedEntity() const -> std::optional<Entity> { return loaded_entity_; }

private:
    std::optional<Entity> loaded_entity_{std::nullopt};
    Image original_img_;
    Image threshold_img_;
    ImageSize img_size_{0, 0, 1};
};

class TransformedImageWindow
{
public:

    void Show();

    void Load(const CpuMatrix &cpu_matrix);

private:
    std::optional<Entity> loaded_entity_{std::nullopt};
    Image original_img_{};
    ImageSize img_size_{0, 0, 1};
    bool has_img_ = false;
    int slider_value_ = 1;
};

class DataInputImageWindow
{
public:
    void Show();

    void LoadImages();

    const std::vector<Entity>& GetLoadedEntities() const { return selected_entity_; };

private:
    std::filesystem::path loading_image_path_{R"(E:\Praca inzynierska\HSI images\)"};
    std::filesystem::path envi_header_path_{};
    std::vector<Entity> selected_entity_;
    std::map<std::filesystem::path, bool> selected_files_;
};

struct ThresholdSetting
{
    float threshold;
    int band;
};

[[nodiscard]] CpuMatrix RunImageThreshold(const CpuMatrix& img, ThresholdSetting setting);

class ThresholdPopupWindow
{
public:
    void Show();

    void Load(const CpuMatrix &cpu_matrix);

    void RunThreshold();

    [[nodiscard]] auto GetThresholdSettings() const -> std::optional<ThresholdSetting> { return saved_settings_; };

    [[nodiscard]] auto GetImageSize() const -> ImageSize { return img_size_; };

private:
    Image original_img_;
    Image threshold_img_;
    ImageSize img_size_{0, 0, 1};
    int selected_band_ = 1;
    float threshold_value_ = 0.f;
    std::optional<ThresholdSetting> saved_settings_ = std::nullopt;
};


struct PcaSetting
{
    std::size_t selected_bands;
};


class PcaPopupWindow
{
public:
    void SetMaxBands(std::size_t n) { max_bands_ = n; };

    void Show();

    [[nodiscard]] auto GetPcaSettings() const -> std::optional<PcaSetting> { return saved_settings_; };
private:
    std::size_t max_bands_ = 0;
    int selected_bands_ = 0;
    std::optional<PcaSetting> saved_settings_ = std::nullopt;
};


class StatisticWindow
{
public:
    void Show();

    void Load(Entity entity, std::vector<StatisticalParameters> param);

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::make_nvp("Statistic values", statistics_));
    }

private:
    std::map<Entity, std::vector<StatisticalParameters>> statistics_{};
};


class DataClassificationWindow
{
public:
    void Show();

    void Load(std::vector<Entity> entities);

    [[nodiscard]] std::map<Entity, int> GetClasses() const { return classes_; };

    [[nodiscard]] int GetClassCount() const { return classes_.size(); };

private:
    std::vector<Entity> entities_{};
    std::map<Entity, int> classes_{};
    int class_count_ = 1;
};



class TreeViewWindow
{
public:
    void Show(const Node *root);

private:
    void ShowNode(const Node *root, int input_id, const ImVec2 pos, int dy);

private:
    constexpr static int dx = 160;
    constexpr static int start_dy = 140;
    constexpr static float scale_dy = 0.85;
    int unique_node_id_ = 0;
    int unique_attr_id_ = 0;
    int unique_link_id_ = 0;
};


class MainWindow
{
public:
    explicit MainWindow(HasNameSystem* has_name_system): has_name_system_{has_name_system} {}

    void Show();

    void SaveStatisticValues();

private:
    void RunTrain();

    void ShowPopupsWindow();

    void UpdateThresholdImage();

    void UpdatePcaImage();

    void ImagePreprocessing();

private:
    ImageViewWindow threshold_window_{};
    std::string selected_img_name_{};
    HasNameSystem *has_name_system_;
    TransformedImageWindow pca_transformed_window_{};
    ThresholdPopupWindow threshold_popup_window_{};
    DataInputImageWindow data_input_window_{};
    PcaPopupWindow pca_popup_window_{};
    StatisticWindow statistic_window_{};
    DataClassificationWindow data_classification_window_{};
    TreeViewWindow tree_view_window_{};
    Tree tree_{};
    SVM svm_{};
    std::vector<CpuMatrix> pca_transformed_images_{};
    std::vector<std::vector<StatisticalParameters>> statistical_params_{};
    ResultPCA result_pca_{};
    ImageSize img_size_{};
    std::string_view selected_model_;
    bool has_run_pca_ = false;
    bool has_run_model_ = false;
    bool add_neighbour_bands_ = false;
};



[[nodiscard]] const char* GetAttributeName(std::size_t idx);

#endif //GUI_HPP
