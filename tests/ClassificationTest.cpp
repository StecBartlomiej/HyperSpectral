#include <catch2/catch_test_macros.hpp>

#include "Classification.hpp"

TEST_CASE("Small data test", "[DecisionTree]")
{
    ObjectList data = {
       { 3,     1,     5,     5,    -1},
       { 4,    -4,     5,     0,     5},
       {-4,    -2,    -4,     3,     3},
       { 5,     1,     5,    -4,     5}
    };

    std::vector<uint32_t> obj_class = {1, 0, 1, 0};

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
