#include "Classification.hpp"

#include "Logger.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <Components.hpp>

#include <iostream>
#include <numeric>
#include <random>
#include <fstream>
#include <map>
#include <cereal/archives/json.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>

extern Coordinator coordinator;


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

    for (const auto& obj_iter: object_list)
    {
        attribute_values.push_back(obj_iter[attribute_idx]);
    }
    std::ranges::sort(attribute_values);

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
    TrainNode(root, object_list, object_class, 0);
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

void Tree::TrainNode(Node *root, const ObjectList &object_list, const std::vector<uint32_t> &object_classes, std::size_t depth)
{
    const bool has_all_of = std::all_of(object_classes.begin(), object_classes.end(),
                [first_class=object_classes.front()](const uint32_t class_idx){ return first_class == class_idx; });
    if (has_all_of)
    {
        LOG_INFO("All object have the same class, exiting");
        root->attribute_idx = object_classes.front();
        return;
    }
    if (depth >= max_depth_)
    {
        // Count class
        std::vector<std::size_t> class_count(class_count_, 0);
        for (auto obj_class : object_classes)
        {
            ++class_count[obj_class];
        }
        const auto iter = std::ranges::max_element(class_count);
        root->attribute_idx = std::distance(class_count.begin(), iter);
        LOG_INFO("Depth exceeded max_depth:{}, closing current branch, class count: [{}]", max_depth_, fmt::join(class_count, ", "));
        return;
    }

    root->left = new Node{};
    root->right = new Node{};

    TreeTest best_test{std::numeric_limits<float>::min(), 0, 0};
    std::vector<std::size_t> D(class_count_, 0);

    for (std::size_t attr_idx = 0; attr_idx < attributes_count_; ++attr_idx)
    {

        auto attribute_values = GetSortedAttributeList(object_list, attr_idx);

        // Iterate over all possible thresholds
        for (std::size_t threshold_idx = 1; threshold_idx < attribute_values.size(); ++threshold_idx)
        {
            // Threshold as mean of adjacent values in sorted array
            const float threshold = (attribute_values[threshold_idx - 1] + attribute_values[threshold_idx]) / 2.f;

            /// Split to d1, d2, D, calclate infromation gain
            std::vector<std::size_t> d_yes(class_count_, 0);
            std::vector<std::size_t> d_no(class_count_, 0);
            std::vector<std::size_t> D(class_count_, 0);

            std::size_t count_d_yes = 0, count_d_no = 0;
            const std::size_t count_D = object_classes.size();

            for (std::size_t obj_idx = 0; obj_idx < object_list.size(); ++obj_idx)
            {
                const auto &attr_list = object_list[obj_idx];
                const auto curr_class = object_classes[obj_idx];
                assert(curr_class < class_count_);

                D[curr_class] += 1;
                if (attr_list[attr_idx] <= threshold)
                {
                    d_no[curr_class] += 1;
                    count_d_no += 1;
                }
                else
                {
                    d_yes[curr_class] += 1;
                    count_d_yes += 1;
                }
            }

            float info_D = 0;
            float info_d_yes = 0;
            float info_d_no = 0;

            for (std::size_t class_idx = 0; class_idx < class_count_; ++class_idx)
            {
                const float p = static_cast<float>(D[class_idx]) / static_cast<float>(count_D);
                assert(p <= 1);
                info_D -= p != 0 ? p * log2(p) : 0;

                const float p_d_no = static_cast<float>(d_no[class_idx]) / static_cast<float>(count_d_no);
                info_d_no -= p_d_no != 0 ? p_d_no * log2(p_d_no) : 0;

                const float p_d_yes = static_cast<float>(d_yes[class_idx]) / static_cast<float>(count_d_yes);
                info_d_yes -= p_d_yes != 0 ? p_d_yes * log2(p_d_yes) : 0;

            }
            float d1_d = static_cast<float>(count_d_yes) / static_cast<float>(count_D);
            float d2_d = static_cast<float>(count_d_no) / static_cast<float>(count_D);

            float gain_d1_d2 = (d1_d * info_d_yes) + (d2_d * info_d_no);
            float info_gain = info_D - gain_d1_d2;

            float split_info = -d1_d * log2(d1_d) - d2_d * log2(d2_d);
            float gain_ration = info_gain / split_info;

            assert(gain_ration >= 0 && gain_ration <= 1);

            if (best_test.information_gain < gain_ration)
            {
                best_test.information_gain = gain_ration;
                best_test.threshold = threshold;
                best_test.attribute_idx = attr_idx;
            }
        }
    }

    if (best_test.information_gain == std::numeric_limits<float>::min())
    {
        std::vector<std::size_t> class_count(class_count_, 0);
        for (auto obj_class : object_classes)
        {
            ++class_count[obj_class];
        }
        const auto iter = std::ranges::max_element(class_count);
        root->attribute_idx = std::distance(class_count.begin(), iter);
        LOG_INFO("Cant split node, closing current branch, class count: [{}]", max_depth_, fmt::join(class_count, ", "));
        return;
    }

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

    LOG_INFO("Best test for current node:\n attribute_idx:{}, threshold:{}, gain_ratio:{}, split {}:{}",
        best_test.attribute_idx, best_test.threshold, best_test.information_gain, left_obj_class.size(), right_obj_class.size());

    if (!left_obj.empty())
    {
        LOG_INFO("Running left node");
        TrainNode(root->left, left_obj, left_obj_class, depth + 1);
    }

    if (!right_obj.empty())
    {
        LOG_INFO("Running right node");
        TrainNode(root->right, right_obj, right_obj_class, depth + 1);
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
    assert(x_.size() == y_.size());

    std::vector<uint32_t> class_result;
    class_result.reserve(x.size());

    for (const auto &attr: x)
    {
        float f = 0.f;
        for (std::size_t i = 0; i < x_.size(); ++i)
        {
            f += alpha_[i] * y_[i] * kernel_(x_[i], attr) + b_;
        }

        class_result.push_back(f >= 0 ? 1 : 0);
    }
    return class_result;
}

std::vector<std::vector<std::size_t>> KFoldGeneration(const std::vector<uint32_t> &object_class, uint32_t class_count, uint32_t k_groups)
{
    assert(!object_class.empty());
    assert(k_groups > 1);
    assert(object_class.size() >= 2);

    std::vector<uint32_t> grouped_count(class_count, 0);
    std::vector<std::vector<std::size_t>> indexes{class_count};

    for (std::size_t i = 0; i < object_class.size(); ++i)
    {
        const auto class_id = object_class[i];

        indexes[class_id].push_back(i);

        assert(class_id < class_count);
        grouped_count[class_id] += 1;
    }

    std::size_t objects_count = object_class.size();

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<std::vector<std::size_t>> object_fold_idx(k_groups);

    for (std::size_t i = 0; i < class_count; ++i)
    {
        std::shuffle(indexes[i].begin(), indexes[i].end(), g);
    }

    // 10 groups gets 1/10 size of every class randomly selected
    for (std::size_t group_idx = 0 ; group_idx < k_groups - 1; ++group_idx)
    {
        for (std::size_t class_idx = 0 ; class_idx < class_count; ++class_idx)
        {
            const float round_div = std::round(grouped_count[class_idx] / static_cast<float>(k_groups));
            const std::size_t current_copy_size = static_cast<std::size_t>(round_div);

            const std::size_t start_idx = group_idx * current_copy_size;
            const std::size_t end_idx = start_idx + current_copy_size;

            for (std::size_t i = start_idx; i < end_idx; ++i)
            {
                const auto curr_idx = indexes[class_idx][i];
                object_fold_idx[group_idx].push_back(curr_idx);
            }
        }
        std::ranges::shuffle(object_fold_idx[group_idx], g);
    }

    // Last group can take more or less than any group
    const auto last_group_idx = k_groups - 1;
    for (std::size_t class_idx = 0 ; class_idx < class_count; ++class_idx)
    {
        const float round_div = std::round(grouped_count[class_idx] / static_cast<float>(k_groups));
        const std::size_t current_copy_size = static_cast<std::size_t>(round_div);

        const std::size_t start_idx = last_group_idx * current_copy_size;
        const std::size_t end_idx = indexes[class_idx].size();

        for (std::size_t i = start_idx; i < end_idx; ++i)
        {
            const auto curr_idx = indexes[class_idx][i];
            object_fold_idx[last_group_idx].push_back(curr_idx);
        }
    }
    std::ranges::shuffle(object_fold_idx[last_group_idx], g);


    return object_fold_idx;
}

TrainingTestData GetFold(const std::vector<std::vector<std::size_t>> &folds, const std::vector<Entity> &object_list,
    const std::vector<uint32_t> &object_class, std::size_t test_fold_idx)
{
    const auto test_fold = folds[test_fold_idx];

    std::vector<Entity> training_data{};
    std::vector<uint32_t> training_classes;

    std::vector<Entity> test_data{};
    std::vector<uint32_t> test_classes;

    training_data.reserve(test_fold.size() * 10);
    test_data.reserve(test_fold.size());

    for (auto j = 0; j < folds.size(); ++j)
    {
        const auto curr_fold = folds[j];

        for (auto l = 0; l < curr_fold.size(); ++l)
        {
            const auto idx = curr_fold[l];
            const auto &curr_object = object_list[idx];
            const auto curr_class  = object_class[idx];

            if (j != test_fold_idx)
            {
                training_data.push_back(curr_object);
                training_classes.push_back(curr_class);
            }
            else
            {
                test_data.push_back(curr_object);
                test_classes.push_back(curr_class);
            }
        }
    }

    return TrainingTestData{
        .training_data = training_data,
        .training_classes = training_classes,
        .test_data = test_data,
        .test_classes = test_classes,
    };
}

TrainingTestData SplitData(const std::vector<Entity> &object_list, const std::vector<uint32_t> &object_classes, std::size_t class_count,
    float split_ratio)
{
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<uint32_t> grouped_count(class_count, 0);
    std::vector<std::vector<std::size_t>> indexes{class_count};

    for (std::size_t i = 0; i < object_classes.size(); ++i)
    {
        const auto class_id = object_classes[i];

        indexes[class_id].push_back(i);

        assert(class_id < class_count);
        grouped_count[class_id] += 1;
    }

    for (std::size_t i = 0; i < class_count; ++i)
    {
        std::ranges::shuffle(indexes[i], g);
    }

    TrainingTestData data{};

    std::vector<std::size_t> train_idx{};
    for (const auto &curr_index : indexes)
    {
        const std::size_t end_idx =  std::round(curr_index.size() * split_ratio);

        for (auto k = 0; k < end_idx; ++k)
        {
            const auto idx = curr_index[k];
            train_idx.push_back(idx);
        }
    }
    std::ranges::shuffle(train_idx, g);
    for (auto idx : train_idx)
    {
        data.training_data.push_back(object_list[idx]);
        data.training_classes.push_back(object_classes[idx]);
    }

    std::vector<std::size_t> test_idx{};
    for (const auto& curr_index : indexes)
    {
        const auto end_idx =  std::round(curr_index.size() * split_ratio);

        for (std::size_t k = end_idx; k < curr_index.size(); ++k)
        {
            const auto idx = curr_index[k];
            test_idx.push_back(idx);
        }
    }
    std::ranges::shuffle(test_idx, g);
    for (auto idx : test_idx)
    {
        data.test_data.push_back(object_list[idx]);
        data.test_classes.push_back(object_classes[idx]);
    }

    return data;
}

PatchSplitData SplitData(const std::vector<PatchData> &object_list, const std::vector<uint8_t> &object_classes,
    std::size_t class_count, float split_ratio)
{
       std::random_device rd;
    std::mt19937 g(rd());

    std::vector<uint32_t> grouped_count(class_count, 0);
    std::vector<std::vector<std::size_t>> indexes{class_count};

    for (std::size_t i = 0; i < object_classes.size(); ++i)
    {
        const auto class_id = object_classes[i];

        indexes[class_id].push_back(i);

        assert(class_id < class_count);
        grouped_count[class_id] += 1;
    }

    for (std::size_t i = 0; i < class_count; ++i)
    {
        std::ranges::shuffle(indexes[i], g);
    }

    PatchSplitData data{};

    std::vector<std::size_t> train_idx{};
    for (const auto &curr_index : indexes)
    {
        const std::size_t end_idx =  std::round(curr_index.size() * split_ratio);

        for (auto k = 0; k < end_idx; ++k)
        {
            const auto idx = curr_index[k];
            train_idx.push_back(idx);
        }
    }
    std::ranges::shuffle(train_idx, g);
    for (auto idx : train_idx)
    {
        data.training_data.push_back(object_list[idx]);
        data.training_classes.push_back(object_classes[idx]);
    }

    std::vector<std::size_t> test_idx{};
    for (const auto& curr_index : indexes)
    {
        const auto end_idx =  std::round(curr_index.size() * split_ratio);

        for (std::size_t k = end_idx; k < curr_index.size(); ++k)
        {
            const auto idx = curr_index[k];
            test_idx.push_back(idx);
        }
    }
    std::ranges::shuffle(test_idx, g);
    for (auto idx : test_idx)
    {
        data.test_data.push_back(object_list[idx]);
        data.test_classes.push_back(object_classes[idx]);
    }

    return data;
}


void SaveClassificationResult(const std::vector<Entity> &data, const std::vector<uint32_t> &data_classes, std::ostream &out)
{
    assert(data.size() == data_classes.size());

    cereal::JSONOutputArchive archive(out);
    std::map<std::string, uint32_t> map{};

    for (std::size_t i = 0; i < data.size(); ++i)
    {
        const auto filename = coordinator.GetComponent<FilesystemPaths>(data[i]).img_data.filename().string();
        map[filename] = data_classes[i];
    }
    archive(cereal::make_nvp("Classification_result", map));
}
