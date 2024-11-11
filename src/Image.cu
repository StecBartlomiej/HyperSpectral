#include "Image.hpp"

#include <cassert>
#include <EntityComponentSystem.hpp>
#include <filesystem>
#include <string>

#include <iostream>
#include <map>


extern Coordinator coordinator;




[[nodiscard]] Entity CreateImage(const FilesystemPaths &paths)
{
    auto id = coordinator.CreateEntity();

    coordinator.AddComponent(id, paths);

    const auto opt_envi = LoadEnvi(paths.envi_header);
    if (!opt_envi.has_value())
    {
        const auto file = paths.envi_header.string();
        LOG_ERROR("CreateImage: failed to load ENVI file {}!", file);
        throw std::runtime_error{"Empty envi header"};
    }
    coordinator.AddComponent(id, opt_envi.value());
    coordinator.AddComponent(id, ImageSize{
        opt_envi->samples_per_image,
        opt_envi->lines_per_image,
        opt_envi->bands_number});


    LOG_INFO("Created image id={}", id);
    return id;
}

[[nodiscard]] std::shared_ptr<float> LoadImage(const std::filesystem::path &path, const EnviHeader &envi)
{
    std::ifstream file{path, std::ios_base::binary | std::ios::in};
    assert(file.is_open());
    return LoadImage(file, envi) ;
}

[[nodiscard]] std::shared_ptr<float> LoadImage(std::istream &iss, const EnviHeader &envi)
{
    switch (envi.data_type)
    {
        case DataType::BYTE:
            return LoadImageType<char>(iss, envi);
        case DataType::INT16:
            return LoadImageType<int16_t>(iss, envi);
        case DataType::INT32:
            return LoadImageType<int32_t>(iss, envi);
        case DataType::INT64:
            return LoadImageType<int64_t>(iss, envi);
        case DataType::UINT16:
            return LoadImageType<int16_t>(iss, envi);
        case DataType::UINT32:
            return LoadImageType<int32_t>(iss, envi);
        case DataType::UINT64:
            return LoadImageType<int64_t>(iss, envi);
        case DataType::FLOAT32:
            return LoadImageType<float>(iss, envi);
        case DataType::FLOAT64:
            return LoadImageType<double>(iss, envi);

        case DataType::COMPLEX32:
        case DataType::COMPLEX64:
        default:
            LOG_ERROR("LoadImage unsupported data type: {}", static_cast<int>(envi.data_type));
            return nullptr;
    }
    return nullptr;
}

std::shared_ptr<float> GetImageData(Entity entity)
{
    static std::map<Entity, std::weak_ptr<float>> loaded_img{};

    const auto iter = loaded_img.find(entity);
    if (iter != loaded_img.end() && !iter->second.expired())
    {
        return iter->second.lock();
    }

    const auto &path = coordinator.GetComponent<FilesystemPaths>(entity).img_data;
    const auto &envi = coordinator.GetComponent<EnviHeader>(entity);

    std::shared_ptr<float> ptr = LoadImage(path, envi);
    loaded_img[entity] = ptr;
    return std::move(ptr);
}


void RunPCA(Entity image)
{
    const auto img_size = coordinator.GetComponent<ImageSize>(image);

    const auto data = GetImageData(image);
    const auto ptr = data.get();

    const auto N = img_size.depth;
    const auto M = img_size.width * img_size.height;


    std::vector<float> mean(N, 0);
    for (std::size_t i = 0; i < M; ++i)
    {
        std::vector<float> pixel_values(N, 0);

        for (std::size_t j = 0; j < N; ++j)
        {
            pixel_values[j] = ptr[j * M + i]; // J * M = band, i - pixel position
            mean[j] += pixel_values[j];
            mean[j] /= static_cast<float>(M);
        }
    }

    std::vector<float> covariance_matrix(N * N, 0);
    for (std::size_t i = 0; i < M; ++i)
    {
        std::vector<float> pixel_values(N, 0);
        for (std::size_t j = 0; j < N; ++j)
        {
            pixel_values[j] = ptr[j * M + i]; // J * M = band, i - pixel position
            pixel_values[j] -= mean[j];
        }

        for (std::size_t j = 0; j < N; ++j)
        {
            for (std::size_t k = 0; k < N; ++k)
            {
                covariance_matrix[j * N + k] += pixel_values[k] * pixel_values[j];
                covariance_matrix[j * N + k] /= static_cast<float>(M);
            }
        }
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            std::cout << covariance_matrix[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout.flush();
}

cudaPitchedPtr LoadImageCuda(const EnviHeader &envi, float *data)
{
    auto width = envi.samples_per_image;
    auto height = envi.lines_per_image;
    auto depth = envi.bands_number;

    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
    cudaPitchedPtr cuda_ptr{};
    GpuAssert(cudaMalloc3D(&cuda_ptr, extent));

//    cudaPitchedPtr host_ptr{.ptr=host_data.get(),
//                            .pitch=opt_envi_header->samples_per_image * sizeof(float),
//                            .xsize=opt_envi_header->lines_per_image,
//                            .ysize=opt_envi_header->bands_number
//    };

    auto host_p = make_cudaPitchedPtr(data, width * sizeof(float),
                                      width * sizeof(float), height);

    cudaMemcpy3DParms params = {0};
    params.srcPtr = host_p;
    params.dstPtr = cuda_ptr;
    params.extent = extent;

    params.kind = cudaMemcpyHostToDevice;


    GpuAssert(cudaMemcpy3D(&params));
    return host_p;
}

__device__ float GetElement(const Matrix matrix, std::size_t y, std::size_t x)
{
    return matrix.data[y * matrix.width + x];
}

__device__ void SetElement(const Matrix matrix, std::size_t y, std::size_t x, float value)
{
    matrix.data[y * matrix.width + x] = value;
}

__global__ void Mean(Matrix img, Matrix mean)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < img.width)
    {
        float result = 0.0f;

        // TODO: improve
        for (std::size_t y = 0; y < img.height; ++y)
        {
            result += GetElement(img, y, x);
        }
        result /= static_cast<float>(img.height);
        SetElement(mean, 0, x, result);
    }
}

__global__ void SubtractMean(Matrix img, Matrix mean)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < img.height && x < img.width)
    {
        const float value = GetElement(img, y, x) - GetElement(mean, 0, x);
        SetElement(img, y, x, value);
    }
}


__global__ void MatMulTrans(const Matrix img, const Matrix result)
{
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= result.width || y >= result.height)
        return;

    float value = 0.f;
    for (std::size_t i = 0; i < img.height; ++i)
    {
        // X^T * X
        value += GetElement(img, i, y) * GetElement(img, i, x);
    }
    value /= static_cast<float>(img.height);
    SetElement(result, y, x, value);
}
