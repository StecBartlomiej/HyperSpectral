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

#define UTF8_TO_CHAR(text) (reinterpret_cast<const char*>(text))

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


void LoadOpenGLTexture(float *data, ImageSize size, GLuint texture, std::size_t selected_band);


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

    [[nodiscard]] const std::vector<Entity>& GetLoadedEntities() const { return selected_entity_; };

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

void SettingsPopupWindow();

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

    [[nodiscard]] int GetClassCount() const { return class_count_ + 1; };

private:
    std::vector<Entity> entities_{};
    std::map<Entity, int> classes_{};
    int class_count_ = 1;
};


class LabelPopupWindow
{
public:
    void Show();

    [[nodiscard]] auto GetLabelFile() const { return label_file_; };

    [[nodiscard]] int GetClassCount() const { return class_count_; };

private:
    std::filesystem::path label_file_{};
    int class_count_ = 1;
};

class TreeViewWindow
{
public:
    void Show(const Node *root);

private:
    void ShowNode(const Node *root, int input_id, ImVec2 pos, int dy);

private:
    int dx = 180;
    int leaf_dx = 100;
    int start_dy = 400;
    int leaf_dy = 80;
    float scale_dy = 0.75;
    float start_scale_dy = 1.5;
    int unique_node_id_ = 0;
    int unique_attr_id_ = 0;
    int unique_link_id_ = 0;
};

class SvmViewWindow
{
public:
    void Show();

    void Set(std::vector<float> alpha, float b);

private:
    std::vector<float> alpha_{};
    float b_{};
};

struct Limits
{
    float min;
    float max;
};

struct NormalizationData
{
    Limits mean, variance, skewness, kurtosis;
};

class MainWindow
{
public:
    void Show();

    void SaveTestClassification();

    void RunModels();

    [[nodsicard]] std::vector<uint32_t> RunClassify(const std::vector<Entity> &entities_vec);

    [[nodiscard]] std::vector<uint32_t> GetObjectClasses(const std::vector<Entity> &entities);

private:
    void RunTrain(const std::vector<Entity> &entities_vec);

    void RunTrainDisjoint(const std::vector<Entity> &image);

    [[nodiscard]] ClassificationData RunTrainPreprocessing(const std::vector<PatchData> &patch_positions,
     const std::vector<uint8_t> &patch_label, Entity image);

    [[nodiscard]] ClassificationData RunTrainPreprocessing(const std::vector<PatchLabel> &patch_positions,
         const std::vector<uint8_t> &patch_label, const std::vector<Entity> &images);

    [[nodiscard]] ObjectList RunPreprocessing(const std::vector<PatchData> &patch_positions, Entity image);

    [[nodiscard]] ObjectList RunPreprocessing(const std::vector<PatchLabel> &patch_positions, const std::vector<Entity> &images);

    [[nodiscard]] std::vector<CpuMatrix> RunThresholding(const std::vector<Entity> &entities_vec);

    void ShowPopupsWindow();

    void ShowPixelApproach();

    void ShowObjectApproach();

    void UpdateThresholdImage();

    void UpdatePcaImage();

    void ImagePreprocessing();

    void RunPca(const std::vector<Entity> &entities_vec);

    [[nodiscard]] ObjectList GetNormalizedData(const std::vector<std::vector<StatisticalParameters>> &statistical_params);

private:
    ImageViewWindow threshold_window_{};
    TransformedImageWindow pca_transformed_window_{};
    ThresholdPopupWindow threshold_popup_window_{};
    DataInputImageWindow data_input_window_{};
    DataInputImageWindow test_input_window_{};
    PcaPopupWindow pca_popup_window_{};
    StatisticWindow statistic_window_{};
    DataClassificationWindow data_classification_window_{};
    TreeViewWindow tree_view_window_{};
    SvmViewWindow svm_view_window_{};
    LabelPopupWindow label_popup_window_{};
    Tree tree_{};
    EnsembleSvm ensemble_svm_{};
    ParametersSVM params_svm_{.max_iter = 100000, .C = 100.f, .tau = 1e-3, .gamma=0.1};
    std::vector<CpuMatrix> pca_transformed_images_{};
    std::vector<std::vector<StatisticalParameters>> statistical_params_{};
    ResultPCA result_pca_{};
    ImageSize img_size_{};
    std::vector<NormalizationData> normalization_data_{};
    std::string_view selected_model_;
    std::string selected_img_name_{};
    float sam_threshold_ = 0.100f;
    float disjoint_data_split_ = 0.2f;
    float disjoint_validation_split_ = 0.5f;
    int k_folds_ = 1;
    int approach_type_ = 0;
    bool has_run_pca_ = false;
    bool has_run_model_ = false;
    bool add_neighbour_bands_ = false;
    bool has_disjoint_sampling_ = false;
    bool has_oversample_ = false;
};


void SaveGroundTruth(const std::vector<PatchData> &patches, const std::vector<uint32_t> &class_result, ImageSize size,
    std::string_view file_name);


[[nodiscard]] const char* GetAttributeName(std::size_t idx);

#endif //GUI_HPP
