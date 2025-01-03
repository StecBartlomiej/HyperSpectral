#include <catch2/catch_test_macros.hpp>

#include "Classification.hpp"

TEST_CASE("Small data test", "[DecisionTree]")
{
    ObjectList data = {
        {0, 2},
        {1, 3},
        {2, 4},
    };

    std::vector<uint32_t> obj_class = {1, 1, 0};

    Tree tree{};

    tree.Train(data, obj_class, 2);
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
