#ifndef HYPERSPECTRAL_IMAGE_HPP
#define HYPERSPECTRAL_IMAGE_HPP

#include "Logger.hpp"
#include "EnviHeader.hpp"
#include "Components.hpp"
#include "EntityComponentSystem.hpp"

#include <memory>
#include <filesystem>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <fstream>
#include <concepts>

#include <cereal/types/vector.hpp>

#ifdef DEBUG
#define CudaAssert(code) \
{  \
    if (code != cudaSuccess) \
    { \
        LOG_ERROR("CUDA assert: {} at {}:{}\n", cudaGetErrorString(code), __FILE__, __LINE__); \
        exit(code); \
    } \
}

#define CusolverAssert(code)\
{ \
    if (code != CUSOLVER_STATUS_SUCCESS) \
    { \
        LOG_ERROR("CUSOLVER error {} at {}:{}\n", static_cast<int>(code), __FILE__, __LINE__); \
        exit(code); \
    } \
}
#else
inline void CudaAssert(cudaError_t code) \
{
    if (code != cudaSuccess)
    {
        LOG_ERROR("CUDA assert: {}\n", cudaGetErrorString(code));
        exit(code);
    }
}

inline void CusolverAssert(cusolverStatus_t code)
{
    if (code != CUSOLVER_STATUS_SUCCESS)
    {
        LOG_ERROR("CUSOLVER error {}\n", static_cast<int>(code));
        exit(code);
    }
}
#endif


struct CpuMatrix;

[[nodiscard]] Entity CreateImage(const FilesystemPaths &paths);

[[nodiscard]] std::shared_ptr<float[]> LoadImage(const std::filesystem::path &path, const EnviHeader &envi);

[[nodiscard]] std::shared_ptr<float[]> LoadImage(std::istream &iss, const EnviHeader &envi);

[[nodiscard]] CpuMatrix GetImageData(Entity entity);

template<typename T>
[[nodiscard]] std::shared_ptr<float[]> LoadImageType(std::istream &iss, const EnviHeader &envi)
{
    assert(envi.byte_order == ByteOrder::LITTLE_ENDIAN);

    std::shared_ptr<float[]> host_data{new float[envi.bands_number *
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


/**
* @param height first dimension of matrix
* @param width second dimension of matrix
* @param data pointer to flatten 2D array of size \a height times \a width
*/
struct Matrix
{
    std::size_t bands_height;
    std::size_t pixels_width;
    float *data;
};

[[nodiscard]] CpuMatrix GetCpuMatrix(ImageSize size);

struct CpuMatrix
{
    ImageSize size;
    std::shared_ptr<float[]> data;

    Matrix GetMatrix() const;

    template<class Archive>
    void serialize(Archive & archive)
    {
        const auto [width, height, depth] = size;
        std::vector<float> img_data{data.get(), data.get() + width * height * depth};
        archive(size, CEREAL_NVP(img_data));
    }
};

template<typename Fn, typename ...Args>
concept ReturnsMatrix = std::same_as<std::invoke_result_t<Fn, Args...>, Matrix>;

template<typename Fn, Fn fn, typename... Args>
requires ReturnsMatrix<Fn, Args...>
[[nodiscard]]
CpuMatrix CudaMatrixToCpu(ImageSize size, Args&&... args)
{
    CpuMatrix cpu_matrix{
        size,
        std::shared_ptr<float[]>(new float[size.width * size.height * size.depth])
    };

    Matrix matrix = fn(std::forward<Args>(args)...);

    assert(matrix.data != nullptr);
    assert(matrix.bands_height * matrix.pixels_width == size.width * size.height * size.depth);

    CudaAssert(cudaMemcpy(cpu_matrix.data.get(), matrix.data, sizeof(float) * size.width * size.height * size.depth, cudaMemcpyDeviceToHost));
    cudaFree(matrix.data);

    return std::move(cpu_matrix);
}


/**
 * @brief Calculates mean of each band
 *
 * @param img input matrix.
 * @param mean matrix of computed means with \a mean.height = img.height and \a mean.width = \a 1
 * @return mean
 */
__global__ void Mean(Matrix img, Matrix mean);

/**
 * @brief Subtracts \a mean values from \a img in place.
 * @param img input matrix, write result of substract to \a img.data
 * @param mean input matrix, with size \a mean.heigth = img.height, \a mean.width = \a 1
 * @return img
 */
__global__ void SubtractMean(Matrix img, Matrix mean);


/**
 * @brief Performs piecewise division of values in matrix
 */
__global__ void PieceWiseDivision(Matrix m, float divisor);

/**
 * @brief Computes matrix multiplication of \a img with transposed \a img.
 * @param img input matrix, \a img.data must be not nullptr
 * @param result result of computed matrix multiplication
 * @param data_count count of images that will be processed
 */
__global__ void MatMulTrans(Matrix img, Matrix result);


[[nodiscard]] float KernelRbfThrust(const AttributeList &a1, const AttributeList &a2, float gamma);

struct ResultPCA
{
    CpuMatrix eigenvalues;
    CpuMatrix eigenvectors;
};

/**
* @brief Calculates covariance matrix of input. Width of input matrix must be observations(pixels) and height
* variables.
*
* @param LoadData function returning ptr to image data accessed by idx, must be of size \a height times \a width, and contain at least
* \a data_count of images.
* @param max_height number of bands in one image.
* @param max_width number of pixels in one image.
* @param data_count number of input images
* @return Matrix with size \a height times \a height. Ptr is allocated on device memory and must be freed manually using cudaFree()!
*/
[[nodiscard]] Matrix CovarianceMatrix(std::function<CpuMatrix(std::size_t)> LoadData,
                                      uint32_t max_height, uint32_t max_width, std::size_t data_count);

/**
* @brief performs PCA
* @param LoadData function returning \a data_count 2d flatten arrays with size \a hegith * \a width
* @param max_height number of bands in one image.
* @param max_width number of pixel in one image.
* @param data_count number of input images.
* @result returns eigenvalues sorted in ascending order and eigenvectors
*/
[[nodiscard]] ResultPCA PCA(std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_height, uint32_t max_width, std::size_t data_count);


[[nodsicard]] CpuMatrix ManualThresholding(Matrix img, std::size_t band, float threshold);

[[nodiscard]] std::size_t SumAll(Matrix img);

__global__ void ConcatNeighboursBand(Matrix img, ImageSize old_size, ImageSize new_size);

[[nodiscard]] CpuMatrix AddNeighboursBand(Matrix img, ImageSize size);

[[nodiscard]] CpuMatrix GetObjectFromMask(Matrix img, Matrix mask);

[[nodiscard]] std::vector<CpuMatrix> MatmulPcaEigenvectors(const CpuMatrix &eigenvectors, std::size_t k_bands,
               std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_pixels, std::size_t data_count);

[[nodsicard]] CpuMatrix GetImportantEigenvectors(const CpuMatrix &eigenvectors, std::size_t k_bands);

[[nodiscard]] float SumAllCuda(Matrix data);

struct StatisticalParameters
{
    float mean;
    float variance;
    float skewness;
    float kurtosis;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            CEREAL_NVP(mean),
            CEREAL_NVP(variance),
            CEREAL_NVP(skewness),
            CEREAL_NVP(kurtosis)
            );
    }
};

__global__ void CalculateFourMovements(Matrix img, Matrix result);

/**
 * @brief Calculates \a StatisticalParameter used in classification algorithm
 * @param cpu_img band is result from projection after PCA,
 * @return vector of statistic parameters for each principal component (band) in \a cpu_img
 */
[[nodiscard]] std::vector<StatisticalParameters> GetStatistics(const CpuMatrix& cpu_img);


class ImageLabel
{
public:
    ImageLabel(const std::filesystem::path &file_path, const ImageSize size);

    uint8_t GetLabels(PatchData patch_pos);

private:
    std::vector<uint8_t> image_label_{};
    ImageSize img_size_{};
};


class PatchSystem
{
public:

    [[nodiscard]] std::size_t GetPatchNumbers(ImageSize size);

    [[nodiscard]] CpuMatrix GetPatchImage(int center_x, int center_y) const;

    [[nodiscard]] PatchData GeneratePatch(ImageSize size, std::size_t patch_idx);

    Entity parent_img = 0;

private:
    static constexpr std::size_t S = PatchData::S;
};

#endif //HYPERSPECTRAL_IMAGE_HPP
