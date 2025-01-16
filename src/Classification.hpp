#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include "EntityComponentSystem.hpp"
#include "Components.hpp"

#include "cereal/cereal.hpp"
#include <cereal/types/vector.hpp>


#include <vector>
#include <string>
#include <functional>

inline uint32_t rng_seed = 12345;


struct Node
{
    std::size_t attribute_idx;
    Node *left;
    Node *right;
    float threshold;
};

struct SavedNode
{
    float threshold;
    std::size_t attribute_idx;
    bool is_leaf;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            threshold,
            attribute_idx,
            is_leaf
            );
    }
};

SavedNode ToSavedNode(const Node *node);

Node FromSavedNode(const SavedNode *node);

[[nodsicard]] bool IsLeaf(const Node *node) noexcept;

using AttributeList = std::vector<float>;
using ObjectList = std::vector<AttributeList>;

[[nodiscard]] AttributeList GetAttributes(const ObjectList &object_list, std::size_t object_idx);

[[nodiscard]] std::vector<float> GetSortedAttributeList(const ObjectList &object_list, std::size_t attribute_idx);


[[nodiscard]] std::vector<float> GetSortedAttributeList(const ObjectList &object_list, const std::vector<uint32_t> &obj_class,
    std::size_t attribute_idx);

struct TreeTest
{
    float information_gain;
    float threshold;
    std::size_t attribute_idx;
};


class Tree
{
public:
    ~Tree();
    Tree operator=(const Tree &tree) = delete;

    void Train(const ObjectList &object_list, const std::vector<uint32_t> &object_class, std::size_t class_count);

    [[nodiscard]] std::vector<uint32_t> Classify(const ObjectList &object_list);

    [[nodiscard]] const Node* GetRoot() const { return root; }

    void Pruning(const ObjectList &train_list, const std::vector<uint32_t> &train_class,
                 const ObjectList &validation_list, const std::vector<uint8_t> &validation_class);

    // void Print();

    [[nodiscard]] uint32_t ClassifyObject(const Node *root, const AttributeList &attributes);

    void TrainNode(Node *node, const ObjectList &object_list, const std::vector<uint32_t> &object_classes, std::size_t depth);

    void PrintNode(const std::string &prefix, const Node *node, bool isLeft);

    template<class Archive>
    void save(Archive & archive) const
    {
        auto vec = GetSavedNodes();

        archive(vec);
    }

    template<class Archive>
    void load(Archive & archive)
    {
        std::vector<SavedNode> vec{};
        archive(vec);

        Reconstruct(vec);
    }
    void Reconstruct(const std::vector<SavedNode> &nodes);

    [[nodiscard]] std::vector<SavedNode> GetSavedNodes() const;

private:
    Node *root{};
    std::size_t attributes_count_ = 0;
    std::size_t class_count_ = 0;
    std::size_t max_depth_ = 8;
};

void FreeNodes(Node *node);

void printBT(Node node);

[[nodiscard]] float KernelRbf(const AttributeList &a1, const AttributeList &a2, float gamma);

[[nodiscard]] float KernelLinear(const AttributeList &a1, const AttributeList &a2);



class SVM
{
public:
    using KernelFunction = std::function<float(const AttributeList&, const AttributeList &)>;

    void Train(const ObjectList &x, const std::vector<int> &y, const KernelFunction &kernel, float C, float tau, std::size_t max_iter);

    std::vector<int> Classify(const ObjectList &x);

    std::vector<float> FunctionValue(const ObjectList &x) const;

    [[nodiscard]] const std::vector<float>& GetAlpha() const { return alpha_; }
    [[nodiscard]] float GetB() const { return b_; }

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            CEREAL_NVP(b_),
            CEREAL_NVP(x_),
            CEREAL_NVP(alpha_),
            CEREAL_NVP(alpha_y_)
            );
    }

    friend std::vector<float> CudaSvmFunctionValue(const ObjectList &object_list, const SVM &svm, float gamma);

private:
    float b_{};
    ObjectList x_{};
    KernelFunction kernel_{};
    std::vector<float> alpha_{};
    std::vector<float> alpha_y_{};
};

struct ParametersSVM
{
    std::size_t max_iter;
    float C;
    float tau;
    float gamma;


    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            CEREAL_NVP(max_iter),
            CEREAL_NVP(C),
            CEREAL_NVP(tau),
            CEREAL_NVP(gamma)
            );
    }
};

class EnsembleSvm
{
public:
    void Train(const ObjectList &x, const std::vector<uint32_t> &y);

    [[nodiscard]] std::vector<uint32_t> Classify(const ObjectList &x) const;

    void SetParameterSvm(std::size_t class_count, ParametersSVM parameters);


    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            CEREAL_NVP(svms_),
            CEREAL_NVP(parameters_)
            );
    }

private:
    std::vector<SVM> svms_{};
    ParametersSVM parameters_{};
};

struct TrainingTestData
{
    std::vector<Entity> training_data;
    std::vector<uint32_t> training_classes;
    std::vector<Entity> test_data;
    std::vector<uint32_t> test_classes;
};

struct ClassificationData
{
    ObjectList objects;
    std::vector<uint32_t> classes;
};

struct PatchSplitData
{
    std::vector<PatchData> training_data;
    std::vector<uint8_t> training_classes;
    std::vector<PatchData> test_data;
    std::vector<uint8_t> test_classes;
};

struct PatchLabelSplitData
{
    std::vector<PatchLabel> training_data;
    std::vector<uint8_t> training_classes;
    std::vector<PatchLabel> test_data;
    std::vector<uint8_t> test_classes;
};



[[nodiscard]] auto KFoldGeneration(const std::vector<uint32_t> &object_class, uint32_t class_count, uint32_t k_groups=10) -> std::vector<std::vector<std::size_t>>;

[[nodsicard]] TrainingTestData GetFold(const std::vector<std::vector<std::size_t>> &folds, const std::vector<Entity> &object_list, const std::vector<uint32_t> &object_class, std::size_t test_fold_idx);

[[nodsicard]] TrainingTestData SplitData(const std::vector<Entity> &object_list, const std::vector<uint32_t> &object_classes, std::size_t class_count, float split_ratio);

[[nodsicard]] PatchSplitData SplitData(const std::vector<PatchData> &object_list, const std::vector<uint8_t> &object_classes, std::size_t class_count, float split_ratio);

[[nodsicard]] PatchLabelSplitData SplitData(const std::vector<PatchLabel> &object_list, const std::vector<uint8_t> &object_classes, std::size_t class_count, float split_ratio);

void SaveClassificationResult(const std::vector<Entity> &data, const std::vector<uint32_t> &data_classes, std::ostream &out);


[[nodiscard]] float ScoreF1(const std::vector<uint32_t> &obj_class, const std::vector<uint32_t> &result_class, uint32_t class_count);


void RandomOversampling(ObjectList &object_list, std::vector<uint32_t> &result_class, std::size_t class_count);


#endif //CLASSIFICATION_HPP
