#include "Image.hpp"

#include <cassert>
#include <EntityComponentSystem.hpp>
#include <filesystem>
#include <string>

#include <iostream>
#include <map>
#include <numeric>


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

[[nodiscard]] std::shared_ptr<float[]> LoadImage(const std::filesystem::path &path, const EnviHeader &envi)
{
    std::ifstream file{path, std::ios_base::binary | std::ios::in};
    assert(file.is_open());
    return LoadImage(file, envi) ;
}

[[nodiscard]] std::shared_ptr<float[]> LoadImage(std::istream &iss, const EnviHeader &envi)
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

std::shared_ptr<float[]> GetImageData(Entity entity)
{
    static std::map<Entity, std::weak_ptr<float[]>> loaded_img{};

    const auto iter = loaded_img.find(entity);
    if (iter != loaded_img.end() && !iter->second.expired())
    {
        return iter->second.lock();
    }

    const auto &path = coordinator.GetComponent<FilesystemPaths>(entity).img_data;
    const auto &envi = coordinator.GetComponent<EnviHeader>(entity);

    std::shared_ptr<float[]> ptr = LoadImage(path, envi);
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
    CudaAssert(cudaMalloc3D(&cuda_ptr, extent));

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


    CudaAssert(cudaMemcpy3D(&params));
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

__device__ void AddElement(const Matrix matrix, std::size_t y, std::size_t x, float value)
{
    matrix.data[y * matrix.width + x] += value;
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


__global__ void MatMulTrans(const Matrix img, const Matrix result, int data_count)
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
    value /= static_cast<float>(img.height * data_count);
    AddElement(result, y, x, value);
}

ResultPCA PCA(std::function<std::shared_ptr<float[]>()> LoadData, uint32_t height, uint32_t width, std::size_t data_count)
{
    std::shared_ptr<float[]> shared_ptr = LoadData();
    Matrix img{height, width, nullptr};
    Matrix mean{1, width, nullptr};
    Matrix cov{width, width, nullptr};

    float *d_img = nullptr;
    float *d_img_2 = nullptr;
    float *d_mean = nullptr;
    float *d_cov = nullptr;
    float *d_eigenvectors = nullptr;
    float *d_eigenvalues = nullptr;

    CudaAssert(cudaMalloc(&d_img, height * width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_img_2, height * width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_mean, width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_cov, width * width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_eigenvalues, width * sizeof(float)));

    CudaAssert(cudaMemcpy(d_img, shared_ptr.get(), height * width * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(d_mean, 0, width * sizeof(float)));
    CudaAssert(cudaMemset(d_cov, 0, width * width * sizeof(float)));

    mean.data = d_mean;
    cov.data = d_cov;

    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    dim3 threads_mean{1024};
    dim3 blocks_mean{(height / 1024) + 1};

    dim3 threads_subtract{64, 16};
    dim3 blocks_subtract{(height / 64) + 1, (width / 16) + 1};

    dim3 threads_matmul{64, 16};
    dim3 blocks_matmul{(height / 64) + 1, (width / 16) + 1};

    float *d_to_cpy = d_img;
    for (std::size_t i = 1; i < data_count; ++i)
    {
        cudaStreamSynchronize(stream1);
        img.data = (i & 1) ? d_img : d_img_2;
        d_to_cpy = (i & 1) ? d_img_2 : d_img;

        Mean<<<blocks_mean, threads_mean, 0, stream1>>>(img, mean);
        SubtractMean<<<blocks_subtract, threads_subtract, 0, stream1>>>(img, mean);
        MatMulTrans<<<blocks_matmul, threads_matmul, 0, stream1>>>(img, cov, static_cast<int>(data_count));

        shared_ptr = LoadData();
        CudaAssert(cudaMemcpyAsync(d_to_cpy, shared_ptr.get(), height * width * sizeof(float), cudaMemcpyHostToDevice));
    }

    img.data = d_to_cpy;
    Mean<<<blocks_mean, threads_mean, 0, stream1>>>(img, mean);
    SubtractMean<<<blocks_subtract, threads_subtract, 0, stream1>>>(img, mean);
    MatMulTrans<<<blocks_matmul, threads_matmul, 0, stream1>>>(img, cov, static_cast<int>(data_count));

    cudaStreamSynchronize(stream1);

    // Calculate eigenvalues
    cusolverDnHandle_t handle = nullptr;
    int *dev_info = nullptr;
    int lwork = 0; // size of workspace
    float *d_work = nullptr;
    constexpr cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    const int size = static_cast<int>(cov.height);

    CusolverAssert(cusolverDnCreate(&handle));
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CusolverAssert(cusolverDnSetStream(handle, stream1));

    CudaAssert(cudaMallocAsync(&dev_info, sizeof(int), stream1));

    CusolverAssert(
        cusolverDnSsyevd_bufferSize(handle, jobz, uplo, size, cov.data, size, d_eigenvalues, &lwork) );
    CudaAssert(cudaMalloc(&d_work, sizeof(float) * lwork));

    CusolverAssert(
        cusolverDnSsyevd(handle, jobz, uplo, size, cov.data, size, d_eigenvalues, d_work, lwork, dev_info) );

    int info = 0;
    CudaAssert(cudaMemcpyAsync(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost, stream1));
    CudaAssert(cudaStreamSynchronize(stream1));
    LOG_INFO("PCA: CusolverDnSsyevd info = {}", info);
    if (info < 0)
    {
        LOG_WARN("PCA: {}-th parameter is wrong", -info);
    }
    CudaAssert(cudaFree(d_work));
    CudaAssert(cudaFree(dev_info));


    auto eigenvector = std::make_unique<float[]>(cov.height * cov.width);
    auto eigenvalues = std::make_unique<float[]>(cov.width);

    cudaMemcpy(eigenvector.get(), cov.data, cov.height * cov.width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvalues.get(), d_eigenvalues, cov.width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    CudaAssert(cudaFree(d_img));
    CudaAssert(cudaFree(d_mean));
    CudaAssert(cudaFree(d_cov));
    CudaAssert(cudaFree(d_eigenvectors));
    CudaAssert(cudaFree(d_eigenvalues));

    CusolverAssert(cusolverDnDestroy(handle));
    CudaAssert(cudaStreamDestroy(stream1));

    CudaAssert(cudaDeviceReset());

    return {.eigenvalues = {cov.height, 1, std::move(eigenvalues)},
            .eigenvectors = {cov.height, cov.width, std::move(eigenvector)}};
}

__global__ void Threshold(Matrix img, float threshold, float *mask)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img.width && y < img.height)
    {
        const float value = img.data[y * img.width + x] > threshold ? 1.f : 0.f;
        mask[y * img.width + x] = value;
    }
}

CpuMatrix ManualThresholding(Matrix img, float threshold)
{
    LOG_INFO("Running ManualThresholding with threshold={}", threshold);
    Matrix d_img{img.height, img.width, nullptr};
    float *d_mask = nullptr;

    CudaAssert(cudaMalloc(&d_img.data, img.height * img.width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_mask, img.height * img.width * sizeof(float)));

    CudaAssert(cudaMemcpy(d_img.data, img.data, img.height * img.width * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(d_mask, 0, img.height *  img.width * sizeof(float)));

    dim3 threads_mean{32, 32};
    dim3 blocks_mean{static_cast<unsigned int>(img.width / 32) + 1, static_cast<unsigned int>(img.height / 32) + 1};
    Threshold<<<threads_mean, blocks_mean>>>(d_img, threshold, d_mask);
    CudaAssert(cudaDeviceSynchronize());

    float *mask = new float[img.height * img.width];

    CudaAssert(cudaMemcpy(mask, d_mask, img.height * img.width * sizeof(float), cudaMemcpyDeviceToHost));

    CudaAssert(cudaFree(d_img.data));
    CudaAssert(cudaFree(d_mask));

    return {img.width, img.height, std::unique_ptr<float[]>{mask}};
}