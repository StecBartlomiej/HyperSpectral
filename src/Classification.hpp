#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <vector>
#include <span>
#include <string>
#include <functional>


struct Node
{
    std::size_t attribute_idx;
    float threshold;

    Node *left;
    Node *right;
};

[[nodsicard]] bool IsLeaf(const Node *node) noexcept;

using AttributeList = std::vector<float>;
using ObjectList = std::vector<AttributeList>;

[[nodiscard]] AttributeList GetAttributes(const ObjectList &object_list, std::size_t object_idx);

[[nodiscard]] std::vector<float> GetSortedAttributeList(const ObjectList &object_list, std::size_t attribute_idx);

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

    void Train(const ObjectList &object_list, const std::vector<uint32_t> &object_class, std::size_t class_count);

    std::vector<uint32_t> Classify(const ObjectList &object_list);

    [[nodiscard]] const Node* GetRoot() const { return root; }

    // void Print();

private:
    [[nodiscard]] uint32_t ClassifyObject(const Node *root, const AttributeList &attributes);

    void TrainNode(Node *node, const ObjectList &object_list, const std::vector<uint32_t> &object_classes);

    void PrintNode(const std::string &prefix, const Node *node, bool isLeft);

private:
    Node *root{};
    std::size_t attributes_count_ = 0;
    std::size_t class_count_ = 0;
};

void FreeNodes(Node *node);

void printBT(Node node);

[[nodiscard]] float KernelRbf(const AttributeList &a1, const AttributeList &a2, float gamma);

class SVM
{
public:
    using KernelFunction = std::function<float(const AttributeList&, const AttributeList &)>;

    void Train(const ObjectList &x, const std::vector<uint32_t> &y, const KernelFunction &kernel);

    std::vector<uint32_t> Classify(const ObjectList &x);

private:
    float b_{};
    std::vector<float> alpha_{};
    ObjectList x_{};
    KernelFunction kernel_{};
    std::vector<uint32_t> y_{};
};



#endif //CLASSIFICATION_HPP
