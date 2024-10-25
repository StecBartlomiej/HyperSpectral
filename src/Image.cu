#include "Image.hpp"

#include <cassert>
#include <EntityComponentSystem.hpp>
#include <filesystem>
#include <string>


extern Coordinator coordinator;


static void GpuAssert(cudaError_t code, bool abort=true)
{
    if (code != cudaSuccess)
    {
        LOG_ERROR("GPU assert: {} {} {}\n", cudaGetErrorString(code), __FILE__, __LINE__);
        if (abort) exit(code);
    }
}

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

void AlgorithmPCA::Run()
{
    // auto dim1 = 1;
    // auto dim2 = 1;
    //
    // std::vector<float> covariance_matrix(dim1 * dim1, 0);
    //
    // /// Calculate mean
    // for (std::size_t i = 0; i < dim1; ++i)
    // {
    //     float mean = 0;
    //     for (std::size_t j = 0; j < dim2; ++j)
    //     {
    //         mean += host_data.get()[i * dim2 + j];
    //     }
    //     mean /= pixel_count;
    //
    //     for (std::size_t j = 0; j < dim2; ++j)
    //     {
    //         host_data.get()[i * dim2 + j] -= mean;
    //     }
    // }
    //
    // // Matrix multiplication
    // for (std::size_t i = 0; i < dim1; ++i)
    // {
    //     for (std::size_t j = 0; j < dim2; ++j)
    //     {
    //         for (std::size_t k = 0; k < dim1; ++k)
    //         {
    //             covariance_matrix[i * dim1 + k] += host_data.get()[i * dim2 + j] * host_data.get()[j * dim1 + k];
    //         }
    //     }
    // }
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

