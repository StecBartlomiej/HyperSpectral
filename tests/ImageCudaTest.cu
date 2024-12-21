#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Image.hpp"

#include <chrono>
#include <memory>



TEST_CASE("Calculating mean", "[CUDA]")
{
    float arr[9] = {1,  1,  3,
                    2, -1, 20,
                    3,  1, 31};

    float arr_mean[] = {0, 0, 0};

    Matrix cuda_img{3, 3, nullptr};
    Matrix cuda_mean{3, 1, nullptr};

    cudaMalloc(&cuda_img.data, 9 * sizeof(float));
    cudaMemcpy(cuda_img.data, arr, 9 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_mean.data, 3 * sizeof(float));
    cudaMemset(cuda_mean.data, 0, 3 * sizeof(float));

    dim3 threadsPerBlock(32);
    dim3 numBlocks(1);
    Mean<<<numBlocks, threadsPerBlock>>>(cuda_img, cuda_mean);
    cudaDeviceSynchronize();

    cudaMemcpy(arr_mean, cuda_mean.data, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float result[] = {5 / 3.f, 21 / 3.f, 35 / 3.f};
    for (std::size_t i = 0; i < 3; i++)
    {
        REQUIRE(result[i] == arr_mean[i]);
    }

    cudaFree(cuda_img.data);
    cudaFree(cuda_mean.data);
}


TEST_CASE("Subtract", "[CUDA]")
{
    float arr[9] = {1, 2, 3,
                   1, -1, 1,
                   3, 20, 31};
    float arr_res[9] = {0};
    float subtract_value[] = {6 / 3.f, 1 / 3.f, 54 / 3.f};

    Matrix img{3, 3, nullptr};
    Matrix mean{3, 1, nullptr};

    cudaMalloc(&img.data, 9 * sizeof(float));
    cudaMemcpy(img.data, arr, 9 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&mean.data, 3 * sizeof(float));
    cudaMemcpy(mean.data, subtract_value, 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(1);
    SubtractMean<<<numBlocks, threadsPerBlock>>>(img, mean);
    cudaDeviceSynchronize();

    cudaMemcpy(arr_res, img.data, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < 3; i++)
    {
        for (std::size_t j = 0; j < 3; j++)
            REQUIRE(arr_res[i * 3 + j] == arr[i * 3 + j] - subtract_value[i]);
    }
    cudaFree(img.data);
    cudaFree(mean.data);
}

TEST_CASE("Matmul square matrix", "[CUDA]")
{
    float arr[9] = {1, 2, 3,
                    4, 5, 6,
                    7, 8, 9};

    float covariance[9] = {0, 0, 0,
                           0, 0, 0,
                           0, 0, 0};

    Matrix img{3, 3, nullptr};
    Matrix cov{3, 3, nullptr};

    cudaMalloc(&img.data, 9 * sizeof(float));
    cudaMemcpy(img.data, arr, 9 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&cov.data, 9 * sizeof(float));
    cudaMemset(cov.data, 0, 9 * sizeof(float));


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(1, 1);
    MatMulTrans<<<numBlocks, threadsPerBlock>>>(img, cov);
    cudaDeviceSynchronize();

    cudaMemcpy(covariance, cov.data, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    float result[] = {14.f, 32.f,  50.f, 32.f, 77.f, 122.f, 50.f, 122.f, 194.f};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(covariance[i * 3 + j] == result[i * 3 + j]);
    }
    cudaFree(img.data);
    cudaFree(cov.data);
}

TEST_CASE("MatMaulTrans rectangle matrix", "[CUDA]")
{
    float arr[] = {1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12};

    float covariance[] = {0, 0, 0,
                          0, 0, 0,
                          0, 0, 0};

    float *cuda_img = nullptr;
    float *cuda_cov = nullptr;

    cudaMalloc(&cuda_img, 12 * sizeof(float));
    cudaMemcpy(cuda_img, arr, 12 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_cov, 12 * sizeof(float));
    cudaMemset(cuda_cov, 0, 9 * sizeof(float));

    const Matrix img{3, 4, cuda_img};
    const Matrix cov{3, 3, cuda_cov};

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(1, 1);
    MatMulTrans<<<numBlocks, threadsPerBlock>>>(img, cov);
    cudaDeviceSynchronize();

    cudaMemcpy(covariance, cuda_cov, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    const float result[] = {
         30,  70, 110,
         70, 174, 278,
        110, 278, 446
    };

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(covariance[i * 3 + j] == result[i * 3 + j]);
    }
    cudaFree(cuda_img);
    cudaFree(cuda_cov);
}

TEST_CASE("Covariance matrix", "[CUDA]")
{
    std::shared_ptr<float[]> arr{new float[]{1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12}};

    float covariance[3 * 3] = {0};
    CpuMatrix cpu_matrix{.size = {.width = 4, .height = 1, .depth = 3}, .data = arr};

    auto LoadFunction = [&](std::size_t i) -> CpuMatrix{
        return cpu_matrix;
    };

    Matrix cov = CovarianceMatrix(LoadFunction, 3, 4, 1);

    CudaAssert(cudaMemcpy(covariance, cov.data, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaFree(cov.data));

    const float res[] = {
        5 / 4.f, 5 / 4.f, 5 / 4.f,
        5 / 4.f, 5 / 4.f, 5 / 4.f,
        5 / 4.f, 5 / 4.f, 5 / 4.f
    };

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            REQUIRE(res[i * 3 + j] == covariance[i * 3 + j]);
        }
    }
}

TEST_CASE("Covariance square matrix", "[CUDA]")
{
    std::shared_ptr<float[]> arr{new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}};

    float covariance[9] = {0};

    CpuMatrix cpu_matrix{.size = {.width = 3, .height = 1, .depth = 3}, .data = arr};

    auto LoadFunction = [&](std::size_t i) -> CpuMatrix{
        return cpu_matrix;
    };

    Matrix cov = CovarianceMatrix(LoadFunction, 3, 3, 1);

    CudaAssert(cudaMemcpy(covariance, cov.data, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaFree(cov.data));

    float res[] = {2 / 3.f, 2 / 3.f, 2 / 3.f, 2 / 3.f, 2 / 3.f, 2 / 3.f, 2 / 3.f, 2 / 3.f, 2 / 3.f};

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(res[i * 3 + j] == covariance[i * 3 + j]);
    }

}

TEST_CASE("Covariance larger matrix", "[CUDA]")
{
    std::shared_ptr<float[]> arr{new float[]{
        0.8222,  0.8276,  0.7758,  0.4249,  0.0869,  0.9525,  0.6927,  0.7245,
        0.5953,  0.2643,  0.3314,  0.5216,  0.3755,  0.8431,  0.7166,  0.3442,
        0.7816,  0.6775,  0.6031,  0.8403,  0.1874,  0.9350,  0.8003,  0.0328,
        0.9877,  0.7939,  0.1840,  0.6250,  0.8105,  0.9734,  0.5937,  0.8436,
        0.7883,  0.6818,  0.0875,  0.2552,  0.7232,  0.3725,  0.0972,  0.7703,
        0.8529,  0.6508,  0.3089,  0.9047,  0.6747,  0.3209,  0.5595,  0.5096,
        0.4856,  0.2373,  0.2309,  0.7673,  0.9140,  0.7832,  0.0431,  0.7070,
        0.8738,  0.4774,  0.9092,  0.5627,  0.8210,  0.2122,  0.7576,  0.7912,
        0.3338,  0.9364,  0.9369,  0.8972,  0.5487,  0.9592,  0.6596,  0.3307,
        0.2053,  0.2411,  0.0319,  0.3681,  0.0155,  0.9302,  0.5995,  0.9137,
        0.4944,  0.2091,  0.5936,  0.3377,  0.0947,  0.3657,  0.2270,  0.5767,
        0.0312,  0.2725,  0.0438,  0.6189,  0.5148,  0.3449,  0.7137,  0.2879
    }};

    const static std::size_t height = 12;
    const static std::size_t width = 8;

    float covariance[height * height] = {0};

    CpuMatrix cpu_matrix{.size = {.width = 8, .height = 1, .depth = 12}, .data = arr};

    auto LoadFunction = [&](std::size_t i) -> CpuMatrix{
        return cpu_matrix;
    };

    Matrix cov = CovarianceMatrix(LoadFunction, height, width, 1);

    CudaAssert(cudaMemcpy(covariance, cov.data, height * height * sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaFree(cov.data));

    const float res[height * height] = {
        0.0678,    0.0150,    0.0363,    0.0052,   -0.0091,   -0.0234,   -0.0371,   -0.0195,    0.0123,    0.0402,    0.0240,   -0.0314,
        0.0150,    0.0372,    0.0377,    0.0157,   -0.0190,   -0.0040,    0.0033,   -0.0192,    0.0038,    0.0342,   -0.0024,    0.0139,
        0.0363,    0.0377,    0.0926,   -0.0023,   -0.0451,    0.0051,   -0.0307,   -0.0338,    0.0438,    0.0035,   -0.0038,    0.0047,
        0.0052,    0.0157,   -0.0023,    0.0596,    0.0511,    0.0161,    0.0359,   -0.0221,   -0.0269,    0.0342,   -0.0097,    0.0009,
       -0.0091,   -0.0190,   -0.0451,    0.0511,    0.0802,    0.0214,    0.0368,    0.0053,   -0.0449,   -0.0022,   -0.0034,   -0.0213,
       -0.0234,   -0.0040,    0.0051,    0.0161,    0.0214,    0.0422,    0.0119,    0.0080,   -0.0173,   -0.0239,   -0.0105,    0.0114,
       -0.0371,    0.0033,   -0.0307,    0.0359,    0.0368,    0.0119,    0.0891,   -0.0161,   -0.0144,    0.0174,   -0.0043,    0.0083,
       -0.0195,   -0.0192,   -0.0338,   -0.0221,    0.0053,    0.0080,   -0.0161,    0.0502,   -0.0343,   -0.0394,    0.0108,   -0.0126,
        0.0123,    0.0038,    0.0438,   -0.0269,   -0.0449,   -0.0173,   -0.0144,   -0.0343,    0.0641,   -0.0077,   -0.0084,    0.0076,
        0.0402,    0.0342,    0.0035,    0.0342,   -0.0022,   -0.0239,    0.0174,   -0.0394,   -0.0077,    0.1164,    0.0128,    0.0191,
        0.0240,   -0.0024,   -0.0038,   -0.0097,   -0.0034,   -0.0105,   -0.0043,    0.0108,   -0.0084,    0.0128,    0.0289,   -0.0268,
       -0.0314,    0.0139,    0.0047,    0.0009,   -0.0213,    0.0114,    0.0083,   -0.0126,    0.0076,    0.0191,   -0.0268,    0.0546,
    };

    for (std::size_t i = 0; i < height; ++i)
    {
        for (std::size_t j = 0; j < height; ++j)
            REQUIRE_THAT(res[i * height + j],
                Catch::Matchers::WithinRel(covariance[i * height + j], 0.05f)
                );
    }
}

TEST_CASE("Covariance two input data", "[CUDA]")
{
    std::shared_ptr<float[]> d1{new float[]{0.6787, 0.3922, 0.7060, 0.0462,
                                            0.7577, 0.6555, 0.0318, 0.0971,
                                            0.7431, 0.1712, 0.2769, 0.8235}};

    std::shared_ptr<float[]> d2{new float[]{0.6948, 0.0344, 0.7655, 0.4898,
                                            0.3171, 0.4387, 0.7952, 0.4456,
                                            0.9502, 0.3816, 0.1869, 0.6463}};

    CpuMatrix cpu_matrix_1{.size = {.width = 2, .height = 2, .depth = 3}, .data = d1};
    CpuMatrix cpu_matrix_2{.size = {.width = 4, .height = 1, .depth = 3}, .data = d2};

    std::vector<CpuMatrix> vec_matrix = {cpu_matrix_1, cpu_matrix_2};
    auto LoadFunction = [&](std::size_t i) -> CpuMatrix{
        return vec_matrix[i];
    };

    float covariance[9] = {0};

    Matrix cov = CovarianceMatrix(LoadFunction, 3, 4, 2);

    CudaAssert(cudaMemcpy(covariance, cov.data, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaFree(cov.data));

    float res[] = {0.0766,  0.0221, -0.0064,
                   0.0221,  0.0716, -0.0224,
                  -0.0064, -0.0224,  0.0817};

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(res[i * 3 + j], Catch::Matchers::WithinRel(covariance[i * 3 + j], 0.02f));
    }

}

TEST_CASE("PCA", "[CUDA]")
{

    std::shared_ptr<float[]> d1{new float[]{0.6787, 0.3922, 0.7060, 0.0462,
                                            0.7577, 0.6555, 0.0318, 0.0971,
                                            0.7431, 0.1712, 0.2769, 0.8235}};

    std::shared_ptr<float[]> d2{new float[]{0.6948, 0.0344, 0.7655, 0.4898,
                                            0.3171, 0.4387, 0.7952, 0.4456,
                                            0.9502, 0.3816, 0.1869, 0.6463}};

    CpuMatrix cpu_matrix_1{.size = {.width = 2, .height = 2, .depth = 3}, .data = d1};
    CpuMatrix cpu_matrix_2{.size = {.width = 4, .height = 1, .depth = 3}, .data = d2};

    std::vector<CpuMatrix> vec_matrix = {cpu_matrix_1, cpu_matrix_2};
    auto LoadFunction = [&](std::size_t i) -> CpuMatrix{
        return vec_matrix[i];
    };

    ResultPCA result = PCA(LoadFunction, 3, 4, 2);

    constexpr std::array<float, 3> eigenvalues = {0.0463, 0.0727, 0.1109};

    for (int i = 0; i < 3; ++i)
    {
        REQUIRE_THAT(eigenvalues[i], Catch::Matchers::WithinRel(result.eigenvalues.data[i], 0.02f) );
    }
}

// TEST_CASE("AddNeighboursBand", "[CUDA]")
// {
//
//     int data[9 * 2] = {1, 2, 3,}
//
//     std::vector<std::shared_ptr<float[]>> data = {std::move(d1), std::move(d2)};
//     auto LoadData = [=, i=0]() mutable ->std::shared_ptr<float[]>{ return data[i++]; };
//
//     ResultPCA result = PCA(LoadData, 3, 4, 2);
//
//     constexpr std::array<float, 4> eigenvalues = {0.0021, 0.0237, 0.0769, 0.1120};
//
//     for (int i = 0; i < 4; ++i)
//     {
//         REQUIRE_THAT(eigenvalues[i], Catch::Matchers::WithinRel(result.eigenvalues.data[i], 0.01f));
//     }
// }

TEST_CASE("GetObjectFromMask", "[CUDA]")
{
    float data[3 * 2] = {1, 2, 3,
                         4, 5, 6};

    float m[3] = {1, 0, 1};

    Matrix img{2, 3, data};
    Matrix mask{1, 3, m};

    CpuMatrix c_new_img = GetObjectFromMask(img, mask);

    float result[4] = {1, 3, 4, 6};

    REQUIRE(c_new_img.size.height * c_new_img.size.width == 2);
    REQUIRE(c_new_img.size.depth == img.bands_height);

    for (int i = 0; i < 4; ++i)
    {
        REQUIRE(c_new_img.data[i] == result[i]);
    }

}

TEST_CASE("Thresholding", "[CUDA]")
{
    float data[3 * 2] = {1, 2, 3,
                         4, 5, 6};

    Matrix img{2, 3, data};

    auto cpu_matrix = ManualThresholding(img, 1, 4);
    float mask_result[3] = {0, 1, 1};

    assert(cpu_matrix.size.width == 3);
    assert(cpu_matrix.size.height == 1);
    assert(cpu_matrix.size.depth  == 1);

    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(cpu_matrix.data[i] == mask_result[i]);
    }
}

TEST_CASE("Thresholding + PCA", "[CUDA]")
{
    float data[3 * 2] = {3, 10, -1,
                         100, 7, 6};

    Matrix img{2, 3, data};

    auto cpu_mask = ManualThresholding(img, 0, 2.f);

    Matrix mask = cpu_mask.GetMatrix();
    float mask_result[3] = {1, 1, 0};
    for (auto i = 0; i < 3; ++i)
    {
        REQUIRE(cpu_mask.data[i] == mask_result[i]);
    }

    auto LoadData = [&](std::size_t i) -> CpuMatrix {
        return GetObjectFromMask(img, mask);;
    };

    auto pca_result = PCA(LoadData, 2, 3, 1);

    for (int i = 0; i < 2; ++i)
    {
        printf("%f ", pca_result.eigenvalues.data[i]);
    }
}

TEST_CASE("Statical parameters", "[CUDA]")
{
    std::shared_ptr<float[]> d1{new float[] {0, -3,  5,  1, -2,  1, -1,  3,
                                             2,  1, -2, -3,  4, -2,  2,  1,
                                            -3,  4,  2,  0, -1,  2,  3, -3 }};

    CpuMatrix input_matrix{{8, 1, 3}, std::move(d1)};

    std::vector<StatisticalParameters> vec_result = GetStatistics(input_matrix);

    //                             Mean   Variance  Skewness  Kurtosis
    const float expected[3 * 4] = {0.5000,  6.0000,   0.3827,  2.2344,
                                   0.3750,  5.2344,  -0.0890,  1.7351,
                                   0.5000,  6.2500,  -0.1920,  1.6144};

    for (std::size_t i = 0; i < vec_result.size(); ++i)
    {
        const auto idx = i * 4;

        const auto [mean, variance, skewness, kurtosis] = vec_result[i];

        REQUIRE_THAT(mean,     Catch::Matchers::WithinRel(expected[idx + 0], 0.02f) );
        REQUIRE_THAT(variance, Catch::Matchers::WithinRel(expected[idx + 1], 0.02f) );
        REQUIRE_THAT(skewness, Catch::Matchers::WithinRel(expected[idx + 2], 0.02f) );
        REQUIRE_THAT(kurtosis, Catch::Matchers::WithinRel(expected[idx + 3], 0.02f) );
    }

}

TEST_CASE("PCA image projection", "[CUDA]")
{
    std::shared_ptr<float[]> d1{new float[] {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }};

    CpuMatrix input_matrix{{3, 1, 3}, std::move(d1)};

    std::shared_ptr<float[]> d2{new float[] {
        1, 1, 1,
        2, 2, 2,
    }};
    CpuMatrix eigenvectors{{3, 2, 1}, std::move(d2)};
    static constexpr std::size_t k_bands = 2;

    auto LoadData = [&](std::size_t i) -> CpuMatrix {
        return input_matrix;
    };

    const auto vec_img = MatmulPcaEigenvectors(eigenvectors, k_bands, LoadData, 3, 1);

    const float expected[] = {12, 15, 18, 24, 30, 36};

    REQUIRE(vec_img.size() == 1);
    const auto &result_img = vec_img.front();

    for (std::size_t i = 0; i < 6; ++i)
    {
        REQUIRE(result_img.data[i] == expected[i]);
    }
}

TEST_CASE("Get important eigenvectors", "[GUI]")
{
    std::shared_ptr<float[]> d1{new float[] {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }};

    CpuMatrix input_matrix{{3, 3, 1}, std::move(d1)};

    auto result = GetImportantEigenvectors(input_matrix, 2);

    REQUIRE(result.size.width == 3);
    REQUIRE(result.size.height == 2);
    REQUIRE(result.size.depth == 1);

    for (std::size_t i = 0; i < 6; ++i)
    {
        REQUIRE(result.data[i] == input_matrix.data[i + 3]);
    }
}

TEST_CASE("Thresholding + PCA + projection + statistic_parameters", "[GUI]")
{
    std::shared_ptr<float[]> d1{new float[]{0.6787, 0.3922, 0.7060, 0.0462,
                                            0.7577, 0.6555, 0.0318, 0.0971,
                                            0.7431, 0.1712, 0.2769, 0.8235}};

    std::shared_ptr<float[]> d2{new float[]{0.6948, 0.0344, 0.7655, 0.4898,
                                            0.3171, 0.4387, 0.7952, 0.4456,
                                            0.9502, 0.3816, 0.1869, 0.6463}};

    CpuMatrix cpu_matrix_1{.size = {.width = 2, .height = 2, .depth = 3}, .data = d1};
    CpuMatrix cpu_matrix_2{.size = {.width = 2, .height = 2, .depth = 3}, .data = d2};




    const std::size_t k_bands = 2;

    /// IMAGE PREPROCESSING
    std::vector<CpuMatrix> cpu_img_objects;
    cpu_img_objects.reserve(2);

    const auto mask_1 = ManualThresholding(cpu_matrix_1.GetMatrix(), 1, 0.5);
    const auto mask_2 = ManualThresholding(cpu_matrix_2.GetMatrix(), 1, 0.5);


    /// Object on mask
    auto cpu_object = GetObjectFromMask(cpu_matrix_1.GetMatrix(), mask_1.GetMatrix());
    cpu_img_objects.push_back(cpu_object);

    cpu_object = GetObjectFromMask(cpu_matrix_2.GetMatrix(), mask_2.GetMatrix());
    cpu_img_objects.push_back(cpu_object);


    /// PCA
    auto LoadData = [&](std::size_t i) -> CpuMatrix { return i == 0 ? cpu_matrix_1 : cpu_matrix_2; };
    ResultPCA result_pca_ = PCA(LoadData,  3, 4, 2);


    /// Get most important eigenvectors
    result_pca_.eigenvectors = GetImportantEigenvectors(result_pca_.eigenvectors, k_bands);

    /// TRANSFORMING IMAGE TO PCA RESULT
    const auto pca_transformed_objects = MatmulPcaEigenvectors(result_pca_.eigenvectors, k_bands, LoadData, 4, 2);
    REQUIRE(pca_transformed_objects.size() == 2);

    std::vector<StatisticalParameters> expected_vec = {
        StatisticalParameters{-0.70563614, 0.04992039, -0.6033294, 2.0229032},
        StatisticalParameters{-0.17557111, 0.12215271, 0.8669471, 2.1088192},
        StatisticalParameters{-0.7693257, 0.093409255, 0.15445937, 1.956167},
        StatisticalParameters{-0.24485776, 0.09730937, -1.0491105, 2.252439}
    };

    {
        const auto &pca_object = pca_transformed_objects[0];

        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);

        for (std::size_t i = 0; i < statistic_vector.size(); ++i)
        {
            const auto &row = statistic_vector[i];
            const auto &exp_row = expected_vec[i];

            REQUIRE_THAT(row.mean, Catch::Matchers::WithinRel(exp_row.mean, 0.02f) );
            REQUIRE_THAT(row.variance, Catch::Matchers::WithinRel(exp_row.variance, 0.02f) );
            REQUIRE_THAT(row.skewness, Catch::Matchers::WithinRel(exp_row.skewness, 0.02f) );
            REQUIRE_THAT(row.kurtosis, Catch::Matchers::WithinRel(exp_row.kurtosis, 0.02f) );
        }
    }

    {
        const auto &pca_object = pca_transformed_objects[1];

        std::vector<StatisticalParameters> statistic_vector = GetStatistics(pca_object);

        for (std::size_t i = 0; i < statistic_vector.size(); ++i)
        {
            const auto &row = statistic_vector[i];
            const auto &exp_row = expected_vec[i + 2];

            REQUIRE_THAT(row.mean, Catch::Matchers::WithinRel(exp_row.mean, 0.02f) );
            REQUIRE_THAT(row.variance, Catch::Matchers::WithinRel(exp_row.variance, 0.02f) );
            REQUIRE_THAT(row.skewness, Catch::Matchers::WithinRel(exp_row.skewness, 0.02f) );
            REQUIRE_THAT(row.kurtosis, Catch::Matchers::WithinRel(exp_row.kurtosis, 0.02f) );
        }
    }

}


TEST_CASE("Mapping new pixel to old pixel", "[CUDA]")
{
    float old_img[] = {
         1,  2,  3,  4,  5, // Depth 1
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,

         1,  2,  3,  4,  5, // Depth 2
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    };
    ImageSize old_size = {5, 4, 2};
    ImageSize new_size = {5-2, 4-2, 2 + 2 * 8};

    Matrix mat{.bands_height = 2, .pixels_width = 20, .data = old_img};

    const CpuMatrix cpu_mat = CudaMatrixToCpu<decltype(&AddNeighboursBand), &AddNeighboursBand>(new_size, mat, old_size);
    REQUIRE(cpu_mat.data != nullptr);

    float expected_img[] = {
        // Mid center
         7,  8,  9, 12, 13, 14, // Band 1
         7,  8,  9, 12, 13, 14, // Band2
        // Up left
         1,  2,  3,  6,  7,  8,
         1,  2,  3,  6,  7,  8,
        // Up mid
         2,  3,  4,  7,  8,  9,
         2,  3,  4,  7,  8,  9,
        // Up right
         3,  4,  5,  8,  9,  10,
         3,  4,  5,  8,  9,  10,
        // Mid left
        6, 7, 8,  11, 12, 13,
        6, 7, 8,  11, 12, 13,
        // Mid right
        8, 9, 10, 13, 14, 15,
        8, 9, 10, 13, 14, 15,
        // Down left
        11, 12, 13, 16, 17, 18,
        11, 12, 13, 16, 17, 18,
        // Down mid
         12, 13, 14, 17, 18, 19,
         12, 13, 14, 17, 18, 19,
        // Down right
        13, 14, 15, 18, 19, 20,
        13, 14, 15, 18, 19, 20
    };

    for (std::size_t i = 0; i < new_size.width * new_size.height * new_size.depth; ++i)
    {
        // printf("%f\n", cpu_mat.data[i]);
        REQUIRE(cpu_mat.data[i] == expected_img[i]);
    }

}
