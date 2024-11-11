#ifndef HYPERSPECTRAL_IMAGE_HPP
#define HYPERSPECTRAL_IMAGE_HPP

#include "Logger.hpp"
#include "EnviHeader.hpp"
#include "Components.hpp"
#include "EntityComponentSystem.hpp"

#include <memory>
#include <filesystem>
#include <cuda_runtime.h>
#include <fstream>


[[nodiscard]] Entity CreateImage(const FilesystemPaths &paths);

[[nodiscard]] std::shared_ptr<float> LoadImage(const std::filesystem::path &path, const EnviHeader &envi);

[[nodiscard]] std::shared_ptr<float> LoadImage(std::istream &iss, const EnviHeader &envi);

[[nodiscard]] std::shared_ptr<float> GetImageData(Entity entity);


void RunPCA(Entity image);

[[nodiscard]] cudaPitchedPtr LoadImageCuda(const EnviHeader &envi, float* data);

template<typename T>
[[nodiscard]] std::shared_ptr<float> LoadImageType(std::istream &iss, const EnviHeader &envi)
{
    assert(envi.byte_order == ByteOrder::LITTLE_ENDIAN);

    std::shared_ptr<float> host_data{new float[envi.bands_number *
                                       envi.lines_per_image *
                                       envi.samples_per_image]};

    std::function<float*(std::size_t, std::size_t, std::size_t)> access_scheme;

    std::size_t dim1, dim2, dim3;
    switch (envi.interleave)
    {
        case Interleave::BSQ:
            dim1 = envi.bands_number;
            dim2 = envi.lines_per_image;
            dim3 = envi.samples_per_image;
            access_scheme = [&, lines_samples=dim2 * dim3, samples=dim3](std::size_t i, std::size_t j, std::size_t k) -> float* {
                return host_data.get() + i * lines_samples  + j * samples + k;
            };
            break;
        case Interleave::BIP:
            dim1 = envi.lines_per_image;
            dim2 = envi.samples_per_image;
            dim3 = envi.bands_number;
            access_scheme = [&, lines_samples=dim2 * dim3, samples=dim3](std::size_t i, std::size_t j, std::size_t k) -> float* {
                return host_data.get() + k * lines_samples + i * samples + j;
            };
            break;
        case Interleave::BIL:
            dim1 = envi.lines_per_image;
            dim2 = envi.bands_number;
            dim3 = envi.samples_per_image;
            access_scheme = [&, lines_samples=dim2 * dim3, samples=dim3](std::size_t i, std::size_t j, std::size_t k) -> float* {
                return host_data.get() + j * lines_samples + i * samples + k;
            };
            break;
    }

    // TODO: add bit order
    T value{};
    for (std::size_t i = 0; i < dim1; ++i)
    {
        for (std::size_t j = 0; j < dim2; ++j)
        {
            for (std::size_t k = 0; k < dim3; ++k)
            {
                iss.read(reinterpret_cast<char*>(&value), sizeof(T));
                *access_scheme(i, j, k) = static_cast<float>(value);
            }
        }
    }
    return std::move(host_data);
}

inline void GpuAssert(cudaError_t code, bool abort=true)
{
    if (code != cudaSuccess)
    {
        LOG_ERROR("GPU assert: {} {} {}\n", cudaGetErrorString(code), __FILE__, __LINE__);
        if (abort) exit(code);
    }
}


/**
* @param height first dimension of matrix
* @param width second dimension of matrix
* @param data pointer to flatten 2D array of size \a height times \a width
*/
struct Matrix
{
    std::size_t height;
    std::size_t width;
    float *data;
};

void PCA(const std::vector<Entity> &images);

/**
 * @brief Calculates mean of each \a M = \a img.height and \a N = \a img.width:
 * \f[ m = \frac{1}{M} \sum_{i=1}^M [x_1 x_2 .. x_n]^T \f]
 *
 * @param img input matrix.
 * @param mean matrix of computed means with \a mean.height = 1 and \a mean.width = \a img.width.
 * @return mean
 */
__global__ void Mean(Matrix img, Matrix mean);

/**
 * @brief Subtracts \a mean values from \a img in place.
 * @param img input matrix, write result of substract to \a img.data
 * @param mean input matrix, with size \a mean.heigth = 1, \a mean.width = \a img.width.
 * @return img
 */
__global__ void SubtractMean(Matrix img, Matrix mean);

__global__ void MatMulTrans(Matrix img, Matrix result);


void CovarianceMatrix(Matrix host_img, Matrix covariance);

#endif //HYPERSPECTRAL_IMAGE_HPP
