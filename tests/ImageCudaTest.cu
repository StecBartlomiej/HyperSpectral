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
    Matrix cuda_mean{1, 3, nullptr};

    cudaMalloc(&cuda_img.data, 9 * sizeof(float));
    cudaMemcpy(cuda_img.data, arr, 9 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_mean.data, 3 * sizeof(float));
    cudaMemset(cuda_mean.data, 0, 3 * sizeof(float));


    auto start = std::chrono::high_resolution_clock::now();
    dim3 threadsPerBlock(32);
    dim3 numBlocks(1);
    Mean<<<numBlocks, threadsPerBlock>>>(cuda_img, cuda_mean);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();


    cudaMemcpy(arr_mean, cuda_mean.data, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float result[] = {6 / 3.f, 1 / 3.f, 54 / 3.f};
    for (std::size_t i = 0; i < 3; i++)
    {
        REQUIRE(result[i] == arr_mean[i]);
    }

    LOG_INFO("Mean kernel execution time: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

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
    Matrix mean{1, 3, nullptr};

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
            REQUIRE(arr_res[i * 3 + j] == arr[i * 3 + j] - subtract_value[j]);
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
    MatMulTrans<<<numBlocks, threadsPerBlock>>>(img, cov, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(covariance, cov.data, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    float result[] = {66 / 3.f, 78 / 3.f, 90 / 3.f, 78 / 3.f, 93 / 3.f, 108 / 3.f, 90 / 3.f, 108 / 3.f, 126 / 3.f};
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

    float covariance[] = {0, 0, 0, 0,
                          0, 0, 0, 0,
                          0, 0, 0, 0};

    float *cuda_img = nullptr;
    float *cuda_cov = nullptr;

    cudaMalloc(&cuda_img, 12 * sizeof(float));
    cudaMemcpy(cuda_img, arr, 12 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_cov, 12 * sizeof(float));
    cudaMemset(cuda_cov, 0, 12 * sizeof(float));

    const Matrix img{3, 4, cuda_img};
    const Matrix cov{3, 4, cuda_cov};

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(1, 1);
    MatMulTrans<<<numBlocks, threadsPerBlock>>>(img, cov, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(covariance, cuda_cov, 12 * sizeof(float), cudaMemcpyDeviceToHost);

    float result[] = {107, 122, 137, 152,
                      122, 140, 158, 176,
                      137, 158, 179, 200,
                      152, 176, 200, 224 };

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE(covariance[i * 4 + j] == result[i * 4 + j] / 3.f);
    }
    cudaFree(cuda_img);
    cudaFree(cuda_cov);
}

void CovarianceMatrix(Matrix host_img, Matrix covariance)
{
    assert(host_img.data != nullptr);
    assert(covariance.data != nullptr);
    assert(covariance.width == host_img.width && covariance.height == host_img.width);

    const std::size_t dim1 = host_img.height;
    const std::size_t dim2 = host_img.width;

    Matrix img{dim1, dim2, nullptr};
    Matrix mean{1, dim2, nullptr};
    Matrix result{dim2, dim2, nullptr};

    CudaAssert(cudaMalloc(&img.data, dim1 * dim2 * sizeof(float)));
    CudaAssert(cudaMemcpy(img.data, host_img.data, dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));

    CudaAssert(cudaMalloc(&mean.data, dim2 * sizeof(float)));
    CudaAssert(cudaMemset(mean.data, 0, dim2 * sizeof(float)));

    CudaAssert(cudaMalloc(&result.data, dim2 * dim2 * sizeof(float)));
    CudaAssert(cudaMemset(result.data, 0, dim2 * dim2 * sizeof(float)));

    dim3 threadsPerBlock1(1024);
    dim3 numBlocks1((dim1 / 1024) + 1);
    Mean<<<numBlocks1, threadsPerBlock1>>>(img, mean);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock2(64, 16);
    dim3 numBlocks2((dim1 / 64) + 1, (dim2 / 16) + 1);
    SubtractMean<<<numBlocks2, threadsPerBlock2>>>(img, mean);

    dim3 threadsPerBlock3(64, 16);
    dim3 numBlocks3((dim1 / 64) + 1, (dim2 / 16) + 1);
    MatMulTrans<<<numBlocks3, threadsPerBlock3>>>(img, result, 1);

    CudaAssert(cudaMemcpy(covariance.data, result.data, dim2 * dim2 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(img.data);
    cudaFree(mean.data);
    cudaFree(result.data);
}

TEST_CASE("Covariance matrix", "[CUDA]")
{
    float arr[] = {1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12};

    float covariance[4 * 4] = {0};


    const std::size_t dim1 = 3;
    const std::size_t dim2 = 4;

    Matrix host_data{dim1, dim2, arr};
    Matrix cov{dim2, dim2, covariance};

    CovarianceMatrix(host_data, cov);

    const float res[] = {
        32 / 3.f, 32 / 3.f, 32 / 3.f, 32 / 3.f,
        32 / 3.f, 32 / 3.f, 32 / 3.f, 32 / 3.f,
        32 / 3.f, 32 / 3.f, 32 / 3.f, 32 / 3.f,
        32 / 3.f, 32 / 3.f, 32 / 3.f, 32 / 3.f
    };

    for (std::size_t i = 0; i < dim2; ++i)
    {
        for (std::size_t j = 0; j < dim2; ++j)
            REQUIRE(res[i * dim2 + j] == covariance[i * dim2 + j]);
    }
}

TEST_CASE("Covariance square matrix", "[CUDA]")
{
    float arr[] = {1, 2, 3,
                   4, 5, 6,
                   7, 8, 9};

    float covariance[3 * 3] = {0};

    const std::size_t dim1 = 3;
    const std::size_t dim2 = 3;

    Matrix host_img{3, 3, arr};
    Matrix cov{3, 3, covariance};

    CovarianceMatrix(host_img, cov);

    float res[] = {6, 6, 6, 6, 6, 6, 6, 6, 6};

    for (std::size_t i = 0; i < dim2; ++i)
    {
        for (std::size_t j = 0; j < dim2; ++j)
            REQUIRE(res[i * dim2 + j] == covariance[i * dim2 + j]);
    }

}

TEST_CASE("Covariance larger matrix", "[CUDA]")
{
    float arr[12 * 8] = {
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
    };

    float covariance[8 * 8] = {0};

    const std::size_t dim1 = 12;
    const std::size_t dim2 = 8;

    Matrix host_data{dim1, dim2, arr};
    Matrix cov{dim2, dim2, covariance};

    CovarianceMatrix(host_data, cov);

    const float res[8 * 8] = {
        0.9812 / 12.f,  0.4613 / 12.f,  0.2974 / 12.f,  0.0274 / 12.f,  0.3128 / 12.f, -0.0172 / 12.f,  0.0182 / 12.f,  0.1990 / 12.f,
        0.4613 / 12.f,  0.7939 / 12.f,  0.4232 / 12.f,  0.2147 / 12.f,  0.1472 / 12.f,  0.2842 / 12.f,  0.2072 / 12.f, -0.0558 / 12.f,
        0.2974 / 12.f,  0.4232 / 12.f,  1.2111 / 12.f,  0.2046 / 12.f, -0.1442 / 12.f,  0.0723 / 12.f,  0.3383 / 12.f, -0.2154 / 12.f,
        0.0274 / 12.f,  0.2147 / 12.f,  0.2046 / 12.f,  0.5470 / 12.f,  0.2802 / 12.f,  0.1394 / 12.f,  0.1977 / 12.f, -0.3624 / 12.f,
        0.3128 / 12.f,  0.1472 / 12.f, -0.1442 / 12.f,  0.2802 / 12.f,  1.1321 / 12.f, -0.3487 / 12.f, -0.2750 / 12.f,  0.1939 / 12.f,
       -0.0172 / 12.f,  0.2842 / 12.f,  0.0723 / 12.f,  0.1394 / 12.f, -0.3487 / 12.f,  1.0542 / 12.f,  0.2167 / 12.f, -0.0692 / 12.f,
        0.0182 / 12.f,  0.2072 / 12.f,  0.3383 / 12.f,  0.1977 / 12.f, -0.2750 / 12.f,  0.2167 / 12.f,  0.7619 / 12.f, -0.3106 / 12.f,
        0.1990 / 12.f, -0.0558 / 12.f, -0.2154 / 12.f, -0.3624 / 12.f,  0.1939 / 12.f, -0.0692 / 12.f, -0.3106 / 12.f,  0.8048 / 12.f
    };

    for (std::size_t i = 0; i < dim2; ++i)
    {
        for (std::size_t j = 0; j < dim2; ++j)
            REQUIRE_THAT(res[i * dim2 + j],
                Catch::Matchers::WithinRel(covariance[i * dim2 + j], 0.01f)
                );
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

    std::vector<std::shared_ptr<float[]>> data = {std::move(d1), std::move(d2)};
    auto LoadData = [=, i=0]() mutable ->std::shared_ptr<float[]>{ return data[i++]; };

    ResultPCA result = PCA(LoadData, 3, 4, 2);

    constexpr std::array<float, 4> eigenvalues = {0.0021, 0.0237, 0.0769, 0.1120};

    for (int i = 0; i < 4; ++i)
    {
        REQUIRE_THAT(eigenvalues[i], Catch::Matchers::WithinRel(result.eigenvalues.data[i], 0.01f));
    }
}
