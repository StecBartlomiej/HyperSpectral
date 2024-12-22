#include "Classification.hpp"

#include "Logger.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <numeric>
#include <crt/host_defines.h>

#include "spdlog/pattern_formatter.h"


bool IsLeaf(const Node *node) noexcept
{
    return node->left == nullptr && node->right == nullptr;
}

AttributeList GetAttributes(const ObjectList &object_list, std::size_t object_idx)
{
    assert(object_idx < object_list.size());
    return object_list[object_idx];
}

std::vector<float> GetSortedAttributeList(const ObjectList &object_list, std::size_t attribute_idx)
{
    std::vector<float> attribute_values{};
    attribute_values.reserve(object_list.size());

    for (const auto obj_iter: object_list)
    {
        attribute_values.push_back(obj_iter[attribute_idx]);
    }
    std::sort(attribute_values.begin(), attribute_values.end());

    return attribute_values;
}

Tree::~Tree()
{
    FreeNodes(root);
}

void Tree::Train(const ObjectList &object_list, const std::vector<uint32_t> &object_class, std::size_t class_count)
{
    assert(!object_list.empty());
    assert(object_class.size() == object_list.size());

    class_count_ = class_count;
    attributes_count_ = object_list.front().size();

    FreeNodes(root);

    root = new Node{};

    LOG_INFO("Training decision tree");
    TrainNode(root, object_list, object_class);
    LOG_INFO("Ended training decision tree");
}

std::vector<uint32_t> Tree::Classify(const ObjectList &object_list)
{
    std::vector<uint32_t> identified_class{};
    identified_class.reserve(object_list.size());

    for (const auto &attributes: object_list)
    {
        auto result = ClassifyObject(root, attributes);
        identified_class.push_back(result);
    }

    return identified_class;
}

// void Tree::Print()
// {
//     PrintNode("", root, false);
// }

uint32_t Tree::ClassifyObject(const Node *root, const AttributeList &attributes)
{
    if (IsLeaf(root))
    {
        return root->attribute_idx;
    }

    const auto obj_value = attributes[root->attribute_idx];

    if (obj_value > root->threshold)
    {
        assert(root->right != nullptr);
        return ClassifyObject(root->right, attributes);
    }
    assert(root->left != nullptr);
    return ClassifyObject(root->left, attributes);
}

void Tree::TrainNode(Node *root, const ObjectList &object_list, const std::vector<uint32_t> &object_classes)
{
    const bool has_all_of = std::all_of(object_classes.begin(), object_classes.end(),
                [first_class=object_classes.front()](const uint32_t class_idx){ return first_class == class_idx; });
    if (has_all_of)
    {
        LOG_INFO("All object have the same class, existing");
        root->attribute_idx = object_classes.front();
        return;
    }

    root->left = new Node{};
    root->right = new Node{};

    TreeTest best_test{std::numeric_limits<float>::min(), 0, 0};

    for (std::size_t attr_idx = 0; attr_idx < attributes_count_; ++attr_idx)
    {
        auto attribute_values = GetSortedAttributeList(object_list, attr_idx);

        // Iterate over all possible thresholds
        for (std::size_t threshold_idx = 1; threshold_idx < attribute_values.size(); ++threshold_idx)
        {
            // Threshold as mean of adjacent values in sorted array
            const float threshold = (attribute_values[threshold_idx - 1] + attribute_values[threshold_idx]) / 2.f;

            /// Split to d1, d2, D, calclate infromation gain
            std::vector<std::size_t> d1(class_count_, 0);
            std::vector<std::size_t> d2(class_count_, 0);
            std::vector<std::size_t> D(class_count_, 0);

            std::size_t count_d1 = 0, count_d2 = 0, count_D = object_list.size();

            for (std::size_t obj_idx = 0; obj_idx < object_list.size(); ++obj_idx)
            {
                const auto &attr_list = object_list[obj_idx];
                const auto curr_class = object_classes[obj_idx];
                assert(curr_class < class_count_);

                D[curr_class] += 1;
                if (attr_list[attr_idx] <= threshold)
                {
                    d1[curr_class] += 1;
                    count_d1 += 1;
                }
                else
                {
                    d2[curr_class] += 1;
                    count_d2 += 1;
                }
            }

            float info_D = 0;
            float info_d1 = 0;
            float info_d2 = 0;
            for (std::size_t class_idx = 0; class_idx < class_count_; ++class_idx)
            {
                const float p = static_cast<float>(D[class_idx]) / static_cast<float>(count_D);
                assert(p <= 1);
                info_D -= p != 0 ? p * log2(p) : 0;

                const float p_d1 = static_cast<float>(d1[class_idx]) / static_cast<float>(count_d1);
                info_d1 -= p_d1 != 0 ? p_d1 * log2(p_d1) : 0;

                const float p_d2 = static_cast<float>(d2[class_idx]) / static_cast<float>(count_d2);
                info_d2 -= p_d2 != 0 ? p_d2 * log2(p_d2) : 0;
            }

            float gain_d1_d2 = ((count_d1 / count_D) * info_d1) + ((count_d2 / count_D) * info_d2);
            float info_gain = info_D - gain_d1_d2;


            if (best_test.information_gain < info_gain)
            {
                best_test.information_gain = info_gain;
                best_test.threshold = threshold;
                best_test.attribute_idx = attr_idx;
            }
        }
    }

    LOG_INFO("Best test for current node, attribute_idx:{}, threshold:{}, informatinon_gain:{}",
        best_test.attribute_idx, best_test.threshold, best_test.information_gain);
    root->attribute_idx = best_test.attribute_idx;
    root->threshold = best_test.threshold;


    // Split Value
    ObjectList left_obj;
    std::vector<uint32_t> left_obj_class;

    ObjectList right_obj;
    std::vector<uint32_t> right_obj_class;

    for (std::size_t i = 0; i < object_list.size(); ++i)
    {
        if (object_list[i][root->attribute_idx] <= root->threshold)
        {
            left_obj.push_back(object_list[i]);
            left_obj_class.push_back(object_classes[i]);
        }
        else
        {
            right_obj.push_back(object_list[i]);
            right_obj_class.push_back(object_classes[i]);
        }
    }

    if (!left_obj.empty())
    {
        LOG_INFO("Running left node");
        TrainNode(root->left, left_obj, left_obj_class);
    }

    if (!right_obj.empty())
    {
        LOG_INFO("Running right node");
        TrainNode(root->right, right_obj, right_obj_class);
    }
}

void Tree::PrintNode(const std::string &prefix, const Node *node, bool isLeft)
{
    std::cout << prefix;
    std::cout << (isLeft ? "├──" : "└──" );

    // print the value of the node
    std::cout << node->attribute_idx << " " << node->threshold << std::endl;

    // enter the next tree level - left and right branch

    if (node->left)
    {
        PrintNode(prefix + (isLeft ? "│   " : "    "), node->left, true);
    }

    if (node->right)
    {
        PrintNode(prefix + (isLeft ? "│   " : "    "), node->right, false);
    }
}

void FreeNodes(Node *node)
{
    if (node == nullptr)
    {
        return;
    }

    FreeNodes(node->left);
    FreeNodes(node->right);

    delete node;
}


float KernelRbf(const AttributeList &a1, const AttributeList &a2, const float gamma)
{
    assert(!a1.empty() && !a2.empty());
    assert(a1.size() == a2.size());

    const float l2_squared = std::accumulate(
        a1.data(),
        a1.data() + a1.size(),
        0.0f,
        [&, i=0](float acc, float val) mutable { return acc + std::pow(val - a2[i++], 2); }
    );

    return std::exp(-gamma * l2_squared);
}


void SVM::Train(const ObjectList &x, const std::vector<uint32_t> &y, const KernelFunction &kernel)
{
    const std::size_t n = x.size();
    bool J = true;
    std::vector<float> alpha(x.size(), 0.f);
    float beta = 1;
    const float step = 0.0001f;
    const float limit = 1e-4;

    while (J)
    {
        J = false;

        for (std::size_t i = 0; i < n; ++i)
        {
            float first_term = 0.f;
            float second_term = 0.f;

            for (std::size_t j = 0; j < n; ++j)
            {
                const float common_value = alpha[j] * y[i] * y[j];

                first_term += common_value * kernel(x[i], x[j]);
                second_term += common_value;
            }

            const float partial_derivative = 1 - first_term - beta * second_term;
            alpha[i] += step * partial_derivative;

            if (alpha[i] < 0)
            {
                alpha[i] = 0;
            }
            else if (alpha[i] > 1)
            {
                alpha[i] = 1;
            }
            else if (abs(partial_derivative) > limit)
            {
                J = true;
            }
        }

        float beta_sum = 0.f;
        for (std::size_t i = 0; i < n; ++i)
        {
            beta_sum += alpha[i] * y[i];
        }
        beta += 0.5f * beta_sum * beta_sum;
    }

    std::size_t count_ns = 0;
    float b = 0.f;

    for (std::size_t s = 0; s < n; ++s)
    {
        if (alpha[s] <= 0 || alpha[s] >= 1)
            continue;
        ++count_ns;

        float sum_alpha = 0.f;
        for (std::size_t i = 0; i < n; ++i)
        {
            sum_alpha += alpha[i] * y[i] * kernel(x[s], x[i]);
        }
        b += y[s] - sum_alpha;
    }
    b /= static_cast<float>(count_ns);

    alpha_ = alpha;
    b_ = b;
    x_ = x;
    y_ = y;
    kernel_ = kernel;
}

std::vector<uint32_t> SVM::Classify(const ObjectList &x)
{
    assert(!x.empty());
    assert(!x_.empty() && !y_.empty());
    assert(x.size() == y_.size());

    std::vector<uint32_t> class_result;
    class_result.reserve(x.size());

    for (const auto &attr: x)
    {
        float f = 0.f;
        for (std::size_t i = 0; i < x_.size(); ++i)
        {
            f += alpha_[i] * y_[i] * kernel_(x[i], attr) + b_;
        }

        class_result.push_back(f >= 0 ? 1 : 0);
    }
    return class_result;
}

