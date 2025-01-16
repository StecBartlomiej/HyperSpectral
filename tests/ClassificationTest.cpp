#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "catch2/benchmark/catch_benchmark.hpp"

#include "Classification.hpp"

#include <fstream>
#include "cereal/archives/binary.hpp"

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

    static constexpr std::string_view save_file = "decision_tree.bin";
    {
        std::ofstream file(save_file.data(), std::ios::binary);

        cereal::BinaryOutputArchive archive(file);
        archive(tree);
    }
    {

    std::ifstream file(save_file.data(), std::ios::binary);

    cereal::BinaryInputArchive archive(file);
    archive(tree);
    }
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
    BENCHMARK("RBF kernel x1, x2") {
        return KernelRbf(x1, x2, gamma);
    };


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
    auto result_class = svm.Classify(object_list);

    REQUIRE(result_class.size() == obj_class.size());

    for (std::size_t i = 0; i < obj_class.size(); i++)
    {
        REQUIRE(result_class[i] == obj_class[i]);
    }
}

TEST_CASE("Ensemble SVM", "[CUDA]")
{
    const AttributeList x1{0.8147, 0.9058, 0.1270, 0.9134, 0.6324};
    const AttributeList x2{0.0975, 0.2785, 0.5469, 0.9575, 0.9649};
    const AttributeList x3{0.1576, 0.9706, 0.9572, 0.4854, 0.8003};
    const AttributeList x4{0.1419, 0.4218, 0.9157, 0.7922, 0.9595};
    const AttributeList x5{-1, -1, -1, -1, -1};
    const AttributeList x6{-1.2, -1.2, -1.2, -1.2, -1.2};

    ObjectList object_list{x1, x2, x3, x4, x5, x6};
    std::vector<uint32_t> obj_class{0, 0, 1, 1, 2, 2};

    const ParametersSVM parameters_svm {
        .max_iter = 10000,
        .C = 100,
        .tau = 0.1,
        .gamma = 0.1
    };

    EnsembleSvm ensemble_svm{};
    ensemble_svm.SetParameterSvm(3, parameters_svm);

    ensemble_svm.Train(object_list, obj_class);
    auto result_class = ensemble_svm.Classify(object_list);

    REQUIRE(result_class.size() == obj_class.size());

    for (std::size_t i = 0; i < obj_class.size(); i++)
    {
        REQUIRE(result_class[i] == obj_class[i]);
    }
}

TEST_CASE("Random oversampling", "[Classification]")
{
    ObjectList object_list{
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4}
    };

    std::vector<uint32_t> obj_class{0, 0, 0, 0, 1};

    RandomOversampling(object_list, obj_class, 2);
}


TEST_CASE("RBF kernel", "[Classification]")
{
    constexpr float gamma = 0.1;

    const AttributeList x1{0.8147, 0.9058, 0.1270, 0.9134, 0.6324};
    const AttributeList x2{0.0975, 0.2785, 0.5469, 0.9575, 0.9649};

    const auto y = KernelRbf(x1, x2, gamma);
    REQUIRE_THAT(y, Catch::Matchers::WithinRel(0.8872f, 0.0001f));
    BENCHMARK("RBF kernel x1, x2") {
        return KernelRbf(x1, x2, gamma);
    };


    const AttributeList x3{0.1576, 0.9706, 0.9572, 0.4854, 0.8003};
    const AttributeList x4{0.1419, 0.4218, 0.9157, 0.7922, 0.9595};

    const auto y2 = KernelRbf(x3, x4, gamma);
    REQUIRE_THAT(y2, Catch::Matchers::WithinRel(0.9586f, 0.0001f));
}

// TEST_CASE("F1 score", "[Classification]")
// {
//     std::vector<uint32_t> obj_class{};
//     std::vector<uint32_t> result_class{};
//
//     // True negative
//     for (std::size_t i = 0; i < 55; i++)
//     {
//         obj_class.push_back(10);
//         result_class.push_back(10);
//     }
//     // False positive
//     for (std::size_t i = 0; i < 5; i++)
//     {
//         obj_class.push_back(10);
//         result_class.push_back(0);
//     }
//     // False negative
//     for (std::size_t i = 0; i < 10; i++)
//     {
//         obj_class.push_back(0);
//         result_class.push_back(10);
//     }
//     // True positive
//     for (std::size_t i = 0; i < 30; i++)
//     {
//         obj_class.push_back(0);
//         result_class.push_back(0);
//     }
//
//     const auto f1_score = ScoreF1(obj_class, result_class, 2);
//
//     REQUIRE_THAT(f1_score, Catch::Matchers::WithinRel(0.799f, 0.01f));
// }
