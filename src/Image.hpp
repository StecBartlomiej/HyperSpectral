#ifndef HYPERSPECTRAL_IMAGE_HPP
#define HYPERSPECTRAL_IMAGE_HPP

#include <Classification.hpp>

#include "Logger.hpp"
#include "Parser/EnviHeader.hpp"
#include "Components.hpp"
#include "EntityComponentSystem.hpp"

#include <memory>
#include <filesystem>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <map>

#include <cereal/types/vector.hpp>

#ifndef NDEBUG
[[always_inline]]
__forceinline
#endif
inline void CudaAssert(const cudaError_t code) \
{
    if (code != cudaSuccess)
    {
        LOG_ERROR("CUDA assert {} at {}:{}\n", cudaGetErrorString(code), __FILE__, __LINE__);
        exit(code);
    }
}

#ifndef NDEBUG
[[always_inline]]
__forceinline
#endif
inline void CusolverAssert(const cusolverStatus_t code)
{
    if (code != CUSOLVER_STATUS_SUCCESS)
    {
        LOG_ERROR("CUSOLVER error {} at {}:{}\n", static_cast<int>(code), __FILE__, __LINE__);
        exit(code);
    }
}

struct CpuMatrix;


[[nodiscard]]
Entity CreateImage(const FilesystemPaths &paths);

[[nodiscard]]
std::shared_ptr<float[]> LoadImage(std::istream &iss, const EnviHeader &envi);

[[nodiscard]]
CpuMatrix GetImageData(Entity entity);


// TODO v2: rework as component - delete redundant imagesize
/**
* @param size corresponds as size of flatten \a data
* @param data cuda pointer to flatten 3D array
*/
struct GpuMatrix
{
    ImageSize size;
    float *data;

    [[nodiscard]] __device__
    float& get(std::size_t ch, std::size_t h, std::size_t w) const
    {
        const auto idx = ch * (size.width * size.height) + h* size.width + w;
        return data[idx];
    }

    [[nodiscard]] __device__
    float& get(std::size_t ch, std::size_t pixel) const
    {
        const auto idx = ch * size.height * size.width + pixel;
        return data[idx];
    }

    [[nodiscard]] __host__ __device__
    std::size_t elements() const { return size.width * size.height * size.channel; }

    [[nodiscard]]
    float* begin() const { return data;}

    [[nodiscard]]
    float* end() const { return data + (size.width * size.height * size.channel);}
};

struct CpuMatrix
{
    ImageSize size;
    std::shared_ptr<float[]> data;

    [[nodiscard]] __host__
    float& get(std::size_t ch, std::size_t h, std::size_t w) const
    {
        return data[FlattenIdx(size, ch, h, w)];
    }

    [[nodiscard]] __host__
    std::size_t elements() const { return size.width * size.height * size.channel; }

    // TODO: make it free function - maybe separate file ? The fuck is this
    // template<class Archive>
    // void serialize(Archive & archive)
    // {
    //     const auto [width, height, depth] = size;
    //     std::vector<float> img_path{data.get(), data.get() + width * height * depth};
    //     archive(size, CEREAL_NVP(img_path));
    // }
};

// template<typename Fn, typename ...Args>
// concept ReturnsMatrix = std::same_as<std::invoke_result_t<Fn, Args...>, GpuMatrix>;
//
// template<typename Fn, Fn fn, typename... Args>
// requires ReturnsMatrix<Fn, Args...>
// [[nodiscard]]
// CpuMatrix CudaMatrixToCpu(ImageSize size, Args&&... args)
// {
//     CpuMatrix cpu_matrix{
//         size,
//         std::shared_ptr<float[]>(new float[size.width * size.height * size.channel])
//     };
//
//     GpuMatrix matrix = fn(std::forward<Args>(args)...);
//
//     assert(matrix.data != nullptr);
//     assert(matrix.bands_height * matrix.pixels_width == size.width * size.height * size.channel);
//
//     CudaAssert(cudaMemcpy(cpu_matrix.data.get(), matrix.data, sizeof(float) * size.width * size.height * size.channel, cudaMemcpyDeviceToHost));
//     cudaFree(matrix.data);
//
//     return std::move(cpu_matrix);
// }


/**
 * @brief Calculates mean of each band
 *
 * @param img input matrix.
 * @param mean matrix of computed means with \a mean.height = img.height and \a mean.width = \a 1
 * @return mean
 */
__global__ void Mean(GpuMatrix img, GpuMatrix mean);

/**
 * @brief Subtracts \a mean values from \a img in place.
 * @param img input matrix, write result of substract to \a img.data
 * @param mean input matrix, with size \a mean.heigth = img.height, \a mean.width = \a 1
 * @return img
 */
__global__ void SubtractMean(GpuMatrix img, GpuMatrix mean);


/**
 * @brief Performs piecewise division of values in matrix
 */
__global__ void PieceWiseDivision(GpuMatrix m, float divisor);

/**
 * @brief Computes matrix multiplication of \a img with transposed \a img.
 * @param img input matrix, \a img.data must be not nullptr
 * @param result result of computed matrix multiplication
 * @param data_count count of images that will be processed
 */
__global__ void MatMulTrans(GpuMatrix img, GpuMatrix result);

[[nodiscard]]
CpuMatrix MultiplyMask(CpuMatrix threshold_mask, CpuMatrix segmentation_mask);

[[nodiscard]]
float KernelRbfThrust(const AttributeList &a1, const AttributeList &a2, float gamma);

// TODO: rework -> non public api
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
[[nodiscard]]
GpuMatrix CovarianceMatrix(std::function<CpuMatrix(std::size_t)> LoadData,
                                      uint32_t max_height, uint32_t max_width, std::size_t data_count);

struct ResultPCA
{
    CpuMatrix eigenvalues;
    CpuMatrix eigenvectors;
};

/**
* @brief performs PCA
* @param LoadData function returning \a data_count 2d flatten arrays with size \a hegith * \a width
* @param max_height number of bands in one image.
* @param max_width number of pixel in one image.
* @param data_count number of input images.
* @result returns eigenvalues sorted in ascending order and eigenvectors
*/
[[nodiscard]]
ResultPCA PCA(std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_height, uint32_t max_width, std::size_t data_count);


[[nodsicard]]
CpuMatrix ManualThresholding(CpuMatrix img, std::size_t band, float threshold);


__global__ void ConcatNeighboursBand(GpuMatrix img, ImageSize new_size);

[[nodiscard]]
CpuMatrix AddNeighboursBand(CpuMatrix img);

[[nodiscard]]
CpuMatrix GetObjectFromMask(CpuMatrix img, CpuMatrix mask);

[[nodiscard]]
std::vector<CpuMatrix> MatmulPcaEigenvectors(const CpuMatrix &eigenvectors, ImageSize new_size,
    std::function<CpuMatrix(std::size_t)> LoadData, std::size_t data_count);

[[nodsicard]]
CpuMatrix GetImportantEigenvectors(const CpuMatrix &eigenvectors, std::size_t k_bands);

// TOOD: is it really needed ???
[[nodiscard]]
float SumAllCuda(CpuMatrix data);

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

__global__ void CalculateFourMovements(GpuMatrix img, GpuMatrix result);

/**
 * @brief Calculates \a StatisticalParameter used in classification algorithm
 * @param cpu_img band is result from projection after PCA,
 * @return vector of statistic parameters for each principal component (band) in \a cpu_img
 */
[[nodiscard]]
std::vector<StatisticalParameters> GetStatistics(const CpuMatrix& cpu_img);


// TODO: clean up this mess
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
    PatchSystem(Entity parent_img);

    [[nodiscard]]
    std::size_t GetPatchNumbers(ImageSize size);

    [[nodiscard]]
    CpuMatrix GetPatchImage(int center_x, int center_y) const;

    [[nodiscard]]
    PatchData GeneratePatch(ImageSize size, std::size_t patch_idx);

    const Entity parent_img;

private:
    static constexpr std::size_t S = PatchData::S;
    std::shared_ptr<float[]> img_data_{};
    ImageSize size_{};
};


class PatchSystemMultiImage
{
public:
    PatchSystemMultiImage(const std::vector<Entity> &images);

    PatchSystem &GetPatchSystem(Entity img) { return map_.at(img); };

private:
    std::map<Entity, PatchSystem> map_{};
};



[[nodiscard]]
std::vector<float> CudaSvmFunctionValue(const ObjectList &object_list, const SVM &svm, float gamma);


/**
* @brief HSI segmentation using Spectral Angle Mapper
* @return mask - value of 0 - does not belong in class, 1 - belongs to class of central pixel
*/
[[nodiscard]]
CpuMatrix SegmentationSAM(CpuMatrix img, float radian_threshold);

#endif //HYPERSPECTRAL_IMAGE_HPP
