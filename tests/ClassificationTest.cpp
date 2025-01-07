#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Classification.hpp"

TEST_CASE("Small data test", "[DecisionTree]")
{
    ObjectList data = {
        {0, 2},
        {1, 3},
        {2, 4},
    };

    std::vector<uint32_t> obj_class = {1, 1, 0};
    std::vector<uint8_t> prune_class = {1, 1, 0};


    Tree tree{};

    tree.Train(data, obj_class, 2);
    tree.Pruning(data, obj_class, data, prune_class);
}

TEST_CASE("K-Fold Validation", "[CUDA]")
{
    std::vector<uint32_t> object_class = {
        0, 0, 0, // 3
        1, 1, 1, 1, 1, 1, 1, // 7
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 11
        3, 3, 3, 3, 3, // 5
    };

    auto result = KFoldGeneration(object_class, 4, 3);
}

TEST_CASE("RBF kernel", "[Classification]")
{
    constexpr float gamma = 0.1;

    const AttributeList x1{0.8147, 0.9058, 0.1270, 0.9134, 0.6324};
    const AttributeList x2{0.0975, 0.2785, 0.5469, 0.9575, 0.9649};

    const auto y = KernelRbf(x1, x2, gamma);
    REQUIRE_THAT(y, Catch::Matchers::WithinRel(0.8872f, 0.0001f));


    const AttributeList x3{0.1576, 0.9706, 0.9572, 0.4854, 0.8003};
    const AttributeList x4{0.1419, 0.4218, 0.9157, 0.7922, 0.9595};

    const auto y2 = KernelRbf(x3, x4, gamma);
    REQUIRE_THAT(y2, Catch::Matchers::WithinRel(0.9586f, 0.0001f));
}

TEST_CASE("SVM", "[Classification]")
{
    const AttributeList x1{0.8147, 0.9058, 0.1270, 0.9134, 0.6324};
    const AttributeList x2{0.0975, 0.2785, 0.5469, 0.9575, 0.9649};
    const AttributeList x3{0.1576, 0.9706, 0.9572, 0.4854, 0.8003};
    const AttributeList x4{0.1419, 0.4218, 0.9157, 0.7922, 0.9595};

    ObjectList object_list{x1, x2, x3, x4};
    std::vector<int> obj_class{1, 1, -1, -1};

    const float C = 100;
    const float tau = 0.1;
    const std::size_t max_iter = 10000;
    const float gamma = 0.1;

    auto kernel_func = [=](const AttributeList &a1, const AttributeList &a2) -> float {
        return KernelRbf(a1, a2, gamma);
    };


    SVM svm{};

    svm.Train(object_list, obj_class, kernel_func, C, tau, max_iter);

}

