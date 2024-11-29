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
    return matrix.data[y * matrix.pixels_width + x];
}

__device__ void SetElement(const Matrix matrix, std::size_t y, std::size_t x, float value)
{
    matrix.data[y * matrix.pixels_width + x] = value;
}

__device__ void AddElement(const Matrix matrix, std::size_t y, std::size_t x, float value)
{
    matrix.data[y * matrix.pixels_width + x] += value;
}

__global__ void Mean(Matrix img, Matrix mean)
{
    const auto y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < img.bands_height)
    {
        float result = 0.0f;

        for (std::size_t x = 0; x < img.pixels_width; ++x)
        {
            result += GetElement(img, y, x);
        }
        result /= static_cast<float>(img.pixels_width);
        SetElement(mean, y, 0, result);
    }
}

__global__ void SumRows(Matrix img, Matrix sum)
{
    const auto y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < img.bands_height)
    {
        float result = 0.0f;

        for (std::size_t x = 0; x < img.pixels_width; ++x)
        {
            result += GetElement(img, y, x);
        }
        AddElement(sum, y, 0, result);
    }
}

__global__ void PieceWiseDivision(Matrix m, float divisor)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m.pixels_width && y < m.bands_height)
    {
        float result = GetElement(m, y, x) / divisor;
        SetElement(m, y, x, result);
    }
}

__global__ void SubtractMean(Matrix img, Matrix mean)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < img.bands_height && x < img.pixels_width)
    {
        const float value = GetElement(img, y, x) - GetElement(mean, y, 0);
        SetElement(img, y, x, value);
    }
}


__global__ void MatMulTrans(const Matrix img, const Matrix result)
{
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= result.pixels_width || y >= result.bands_height)
        return;

    float value = 0.f;
    for (std::size_t i = 0; i < img.pixels_width; ++i)
    {
        // X * X^T
        value += GetElement(img, y, i) * GetElement(img, x, i);
    }
    AddElement(result, y, x, value);
}

Matrix CovarianceMatrix(std::function<std::shared_ptr<float[]>(std::size_t)> LoadData, uint32_t height, uint32_t width, std::size_t data_count)
{
    // pixels_width = x = pixels_width, bands_height = y = bands_height

    auto blocking_load_img = [&, height, width](std::size_t i, float *data) {
        const std::shared_ptr<float[]> shared_ptr = LoadData(i);
        CudaAssert(cudaMemcpy(data, shared_ptr.get(), height * width * sizeof(float), cudaMemcpyHostToDevice));
    };

    Matrix img{height, width, nullptr};
    Matrix mean{height, 1, nullptr};
    Matrix cov{height, height, nullptr};

    float *d_to_copy = nullptr;

    CudaAssert(cudaMalloc(&img.data, height * width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_to_copy, height * width * sizeof(float)));
    CudaAssert(cudaMalloc(&mean.data, height * sizeof(float)));
    CudaAssert(cudaMalloc(&cov.data, height * height * sizeof(float)));

    CudaAssert(cudaMemset(mean.data, 0, height * sizeof(float)));
    CudaAssert(cudaMemset(cov.data, 0, height * height * sizeof(float)));


    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));


    dim3 threads_sum{1024};
    dim3 blocks_sum{(height / 1024) + 1};

    dim3 threads_division{1, 1024};
    dim3 blocks_division{1, (height / 1024) + 1};

    dim3 threads_division_2{32, 32};
    dim3 blocks_division_2{(height / 32) + 1, (height / 32) + 1};

    dim3 threads_subtract{64, 16};
    dim3 blocks_subtract{(height / 64) + 1, (width / 16) + 1};

    dim3 threads_matmul{64, 16};
    dim3 blocks_matmul{(height / 64) + 1, (width / 16) + 1};


    blocking_load_img(0, img.data);
    for (std::size_t i = 0; i < data_count - 1; ++i)
    {
        SumRows<<<blocks_sum, threads_sum, 0, stream1>>>(img, mean);

        // Load in parallel
        blocking_load_img(i + 1, d_to_copy);

        cudaStreamSynchronize(stream1);
        std::swap(img.data, d_to_copy);
    }
    SumRows<<<blocks_sum, threads_sum, 0, stream1>>>(img, mean);
    PieceWiseDivision<<<blocks_division, threads_division, 0, stream1>>>(mean, static_cast<float>(img.pixels_width * data_count));


    blocking_load_img(0, img.data);
    for (std::size_t i = 0; i < data_count - 1; ++i)
    {
        SubtractMean<<<blocks_subtract, threads_subtract, 0, stream1>>>(img, mean);
        MatMulTrans<<<blocks_matmul, threads_matmul, 0, stream1>>>(img, cov);

        // Load in parallel
        blocking_load_img(i + 1, d_to_copy);

        cudaStreamSynchronize(stream1);
        std::swap(img.data, d_to_copy);
    }
    SubtractMean<<<blocks_subtract, threads_subtract, 0, stream1>>>(img, mean);
    MatMulTrans<<<blocks_matmul, threads_matmul, 0, stream1>>>(img, cov);
    PieceWiseDivision<<<blocks_division_2, threads_division_2, 0, stream1>>>(cov, static_cast<float>(img.pixels_width * data_count));
    cudaStreamSynchronize(stream1);


    CudaAssert(cudaFree(img.data));
    CudaAssert(cudaFree(mean.data));
    CudaAssert(cudaFree(d_to_copy));

    CudaAssert(cudaStreamDestroy(stream1));

    return cov;
}

ResultPCA PCA(std::function<std::shared_ptr<float[]>(std::size_t)> LoadData, uint32_t height, uint32_t width, std::size_t data_count)
{
    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    Matrix cov = CovarianceMatrix(LoadData, height, width, data_count);

    float *d_eigenvalues = nullptr;
    CudaAssert(cudaMalloc(&d_eigenvalues, height * sizeof(float)));

    // Calculate eigenvalues
    cusolverDnHandle_t handle = nullptr;
    int *dev_info = nullptr;
    int lwork = 0; // size of workspace
    float *d_work = nullptr;
    constexpr cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    const int size = static_cast<int>(cov.bands_height);

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


    auto eigenvector = std::make_unique<float[]>(cov.bands_height * cov.pixels_width);
    auto eigenvalues = std::make_unique<float[]>(cov.pixels_width);

    cudaMemcpy(eigenvector.get(), cov.data, cov.bands_height * cov.pixels_width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvalues.get(), d_eigenvalues, cov.pixels_width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    CudaAssert(cudaFree(d_eigenvalues));

    CusolverAssert(cusolverDnDestroy(handle));
    CudaAssert(cudaStreamDestroy(stream1));

    CudaAssert(cudaDeviceReset());

    return {.eigenvalues = {cov.bands_height, 1, std::move(eigenvalues)},
            .eigenvectors = {cov.bands_height, cov.pixels_width, std::move(eigenvector)}};
}

__global__ void Threshold(Matrix img, float threshold, float *mask)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img.pixels_width && y < img.bands_height)
    {
        const float value = GetElement(img, y, x) > threshold ? 1.f : 0.f;
        mask[y * img.pixels_width + x] = value;
    }
}

CpuMatrix ManualThresholding(Matrix img, float threshold)
{
    LOG_INFO("Running ManualThresholding with threshold={}", threshold);
    Matrix d_img{img.bands_height, img.pixels_width, nullptr};
    float *d_mask = nullptr;

    CudaAssert(cudaMalloc(&d_img.data, img.bands_height * img.pixels_width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_mask, img.bands_height * img.pixels_width * sizeof(float)));

    CudaAssert(cudaMemcpy(d_img.data, img.data, img.bands_height * img.pixels_width * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(d_mask, 0, img.bands_height *  img.pixels_width * sizeof(float)));

    dim3 threads_mean{32, 32};
    dim3 blocks_mean{static_cast<unsigned int>(img.pixels_width / 32) + 1, static_cast<unsigned int>(img.bands_height / 32) + 1};
    Threshold<<<threads_mean, blocks_mean>>>(d_img, threshold, d_mask);
    CudaAssert(cudaDeviceSynchronize());

    float *mask = new float[img.bands_height * img.pixels_width];

    CudaAssert(cudaMemcpy(mask, d_mask, img.bands_height * img.pixels_width * sizeof(float), cudaMemcpyDeviceToHost));

    CudaAssert(cudaFree(d_img.data));
    CudaAssert(cudaFree(d_mask));

    return {img.width, img.height, std::unique_ptr<float[]>{mask}};
}
    return {img.pixels_width, img.bands_height, std::unique_ptr<float[]>{mask}};
}