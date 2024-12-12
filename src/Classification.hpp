#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <vector>
#include <span>


struct Node
{
    std::size_t idx_left;
    std::size_t idx_right;

    std::size_t attribute_idx;
    float threshold;
};

[[nodsicard]] bool IsLeaf(const Node &node) noexcept;

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
    void Train(const ObjectList &object_list, const std::vector<uint32_t> &object_class, std::size_t class_count);

private:
    void TrainNode(Node root, const ObjectList &object_list, const std::vector<uint32_t> &object_classes);

private:
    std::vector<Node> nodes_{};
    std::size_t attributes_count_ = 0;
    std::size_t class_count_ = 0;
};

#endif //CLASSIFICATION_HPP
