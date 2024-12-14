#include "Image.hpp"

#include <cassert>
#include <EntityComponentSystem.hpp>
#include <filesystem>
#include <string>

#include <iostream>
#include <map>
#include <numeric>
#include <span>
#include <cmath>

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
    return LoadImage(file, envi);
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

CpuMatrix GetImageData(Entity entity)
{
    static std::map<Entity, std::weak_ptr<float[]>> loaded_img{};

    const auto &size = coordinator.GetComponent<ImageSize>(entity);

    const auto iter = loaded_img.find(entity);
    if (iter != loaded_img.end() && !iter->second.expired())
    {
        return CpuMatrix{size, iter->second.lock()};
    }

    const auto &path = coordinator.GetComponent<FilesystemPaths>(entity).img_data;
    const auto &envi = coordinator.GetComponent<EnviHeader>(entity);

    std::shared_ptr<float[]> ptr = LoadImage(path, envi);
    loaded_img[entity] = ptr;

    return CpuMatrix{size, std::move(ptr)};
}


void RunPCA(Entity image)
{
    const auto img_size = coordinator.GetComponent<ImageSize>(image);

    const auto data = GetImageData(image).data;
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

Matrix CpuMatrix::GetMatrix() const
{
    return Matrix{
        .bands_height = size.depth,
        .pixels_width = size.width * size.height,
        .data = data.get()
    };
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

Matrix CovarianceMatrix(std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_height, uint32_t max_width, std::size_t data_count)
{
    // pixels_width = x = pixels_width = ImageSize.max_width * ImageSize.hegith, bands_height = y = bands_height= ImageSize.depth

    auto blocking_load_img = [&, max_height, max_width](std::size_t i, Matrix &img) -> ImageSize {
        auto [size, ptr] = LoadData(i);

        img.pixels_width = size.width * size.height;
        img.bands_height = size.depth;

        assert(img.bands_height <= max_height);
        assert(img.pixels_width <= max_width);

        CudaAssert(cudaMemcpy(img.data, ptr.get(), size.height * size.width * size.depth * sizeof(float), cudaMemcpyHostToDevice));
        return size;
    };

    Matrix img{0, 0, nullptr};
    Matrix mean{max_height, 1, nullptr};
    Matrix cov{max_height, max_height, nullptr};

    Matrix img_to_copy{0, 0, nullptr};

    CudaAssert(cudaMalloc(&img.data, max_height * max_width * sizeof(float)));
    CudaAssert(cudaMalloc(&img_to_copy.data, max_height * max_width * sizeof(float)));
    CudaAssert(cudaMalloc(&mean.data, max_height * sizeof(float)));
    CudaAssert(cudaMalloc(&cov.data, max_height * max_height * sizeof(float)));

    CudaAssert(cudaMemset(mean.data, 0, max_height * sizeof(float)));
    CudaAssert(cudaMemset(cov.data, 0, max_height * max_height * sizeof(float)));


    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));


    dim3 threads_sum{1024};
    dim3 blocks_sum{(max_height / 1024) + 1};

    dim3 threads_division{1, 1024};
    dim3 blocks_division{1, (max_height / 1024) + 1};

    dim3 threads_division_2{32, 32};
    dim3 blocks_division_2{(max_height / 32) + 1, (max_height / 32) + 1};

    dim3 threads_subtract{64, 16};
    dim3 blocks_subtract{(max_height / 64) + 1, (max_width / 16) + 1};

    dim3 threads_matmul{64, 16};
    dim3 blocks_matmul{(max_height / 64) + 1, (max_width / 16) + 1};


    blocking_load_img(0, img);
    for (std::size_t i = 0; i < data_count - 1; ++i)
    {
        SumRows<<<blocks_sum, threads_sum, 0, stream1>>>(img, mean);

        // Load in parallel
        blocking_load_img(i + 1, img_to_copy);

        cudaStreamSynchronize(stream1);
        std::swap(img, img_to_copy);
    }
    SumRows<<<blocks_sum, threads_sum, 0, stream1>>>(img, mean);
    PieceWiseDivision<<<blocks_division, threads_division, 0, stream1>>>(mean, static_cast<float>(img.pixels_width * data_count));


    blocking_load_img(0, img);
    for (std::size_t i = 0; i < data_count - 1; ++i)
    {
        SubtractMean<<<blocks_subtract, threads_subtract, 0, stream1>>>(img, mean);
        MatMulTrans<<<blocks_matmul, threads_matmul, 0, stream1>>>(img, cov);

        // Load in parallel
        blocking_load_img(i + 1, img_to_copy);

        cudaStreamSynchronize(stream1);
        std::swap(img, img_to_copy);
    }
    SubtractMean<<<blocks_subtract, threads_subtract, 0, stream1>>>(img, mean);
    MatMulTrans<<<blocks_matmul, threads_matmul, 0, stream1>>>(img, cov);
    PieceWiseDivision<<<blocks_division_2, threads_division_2, 0, stream1>>>(cov, static_cast<float>(img.pixels_width * data_count));
    cudaStreamSynchronize(stream1);


    CudaAssert(cudaFree(img.data));
    CudaAssert(cudaFree(mean.data));
    CudaAssert(cudaFree(img_to_copy.data));

    CudaAssert(cudaStreamDestroy(stream1));

    return cov;
}

ResultPCA PCA(std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_height, uint32_t max_width, std::size_t data_count)
{
    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    Matrix cov = CovarianceMatrix(LoadData, max_height, max_width, data_count);

    float *d_eigenvalues = nullptr;
    CudaAssert(cudaMalloc(&d_eigenvalues, max_height * sizeof(float)));

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


    auto eigenvector = std::make_shared<float[]>(cov.bands_height * cov.pixels_width);
    auto eigenvalues = std::make_shared<float[]>(cov.pixels_width);

    cudaMemcpy(eigenvector.get(), cov.data, cov.bands_height * cov.pixels_width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvalues.get(), d_eigenvalues, cov.pixels_width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    CudaAssert(cudaFree(d_eigenvalues));

    CusolverAssert(cusolverDnDestroy(handle));
    CudaAssert(cudaStreamDestroy(stream1));

    CudaAssert(cudaDeviceReset());

    CpuMatrix mat_eigenvalues{
        .size = ImageSize{
            .width = 1,
            .height = static_cast<uint32_t>(cov.bands_height),
            .depth = 1},
        .data = std::move(eigenvalues)
    };
    CpuMatrix mat_eigenvectors{
        .size = ImageSize{
            .width = static_cast<uint32_t>(cov.pixels_width),
            .height = static_cast<uint32_t>(cov.bands_height),
            .depth = 1},
        .data = std::move(eigenvector)
    };


    return {.eigenvalues = mat_eigenvalues, .eigenvectors = mat_eigenvectors};
}

__global__ void Threshold(Matrix img, std::size_t band, float threshold, float *mask)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < img.pixels_width)
    {
        const float value = GetElement(img, band, x) > threshold ? 1.f : 0.f;
        mask[x] = value;
    }
}

CpuMatrix ManualThresholding(Matrix img, std::size_t band, float threshold)
{
    Matrix d_img{img.bands_height, img.pixels_width, nullptr};
    float *d_mask = nullptr;

    CudaAssert(cudaMalloc(&d_img.data, img.bands_height * img.pixels_width * sizeof(float)));
    CudaAssert(cudaMalloc(&d_mask, img.pixels_width * sizeof(float)));

    CudaAssert(cudaMemcpy(d_img.data, img.data, img.bands_height * img.pixels_width * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(d_mask, 0, img.pixels_width * sizeof(float)));

    dim3 threads_mean{1024};
    dim3 blocks_mean{static_cast<unsigned int>(img.pixels_width) / 1024 + 1};
    Threshold<<<blocks_mean, threads_mean>>>(d_img, band, threshold, d_mask);
    CudaAssert(cudaDeviceSynchronize());

    std::shared_ptr<float[]> mask{new float[img.pixels_width]};

    CudaAssert(cudaMemcpy(mask.get(), d_mask, img.pixels_width * sizeof(float), cudaMemcpyDeviceToHost));

    CudaAssert(cudaFree(d_img.data));
    CudaAssert(cudaFree(d_mask));

    ImageSize img_size = {
        .width = static_cast<uint32_t>(img.pixels_width),
        .height = static_cast<uint32_t>(1),
        .depth = 1};

    return {img_size, std::move(mask)};
}

std::size_t SumAll(Matrix img)
{
    return static_cast<std::size_t>(std::accumulate(img.data, img.data + img.pixels_width + img.pixels_width * (img.bands_height - 1), 0.f));
}

__global__ void ConcatNeighboursBand(Matrix old_img, Matrix new_img)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x < old_img.pixels_width - 2 && y < old_img.bands_height - 2)
    {
        const auto up_left_corn = GetElement(old_img, y - 1, x - 1);
        const auto up_center = GetElement(old_img, y, x - 1);
        const auto up_right_corn = GetElement(old_img, y + 1, x - 1);

        const auto mid_left = GetElement(old_img, y - 1, x);
        const auto mid_center = GetElement(old_img, y, x);
        const auto mid_right = GetElement(old_img, y + 1, x);

        const auto down_left_corn = GetElement(old_img, y - 1, x + 1);
        const auto down_center = GetElement(old_img, y, x + 1);
        const auto down_right_corn = GetElement(old_img, y + 1, x + 1);

        SetElement(new_img, y - 1, x - 1, up_left_corn);
        SetElement(new_img, y - 1, x, up_center);
        SetElement(new_img, y - 1, x + 1, up_right_corn);

        SetElement(new_img, y, x - 1, mid_left);
        SetElement(new_img, y, x, mid_center);
        SetElement(new_img, y, x + 1, mid_right);

        SetElement(new_img, y + 1, x - 1, down_left_corn);
        SetElement(new_img, y + 1, x, down_center);
        SetElement(new_img, y + 1, x + 1, down_right_corn);
    }

}

Matrix AddNeighboursBand(Matrix img)
{
    Matrix new_img{img.bands_height, img.pixels_width * 8, nullptr};
    Matrix old_img{img.bands_height, img.pixels_width, nullptr};

    CudaAssert(cudaMalloc(&old_img.data, old_img.bands_height * old_img.pixels_width * sizeof(float)));
    CudaAssert(cudaMalloc(&new_img.data, new_img.bands_height * new_img.pixels_width * sizeof(float)));

    CudaAssert(cudaMemcpy(old_img.data, img.data, old_img.bands_height * old_img.pixels_width * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads_mean{32, 32};
    dim3 blocks_mean{static_cast<unsigned int>(old_img.bands_height / 32 + 1), static_cast<unsigned int>(old_img.pixels_width / 32 + 1)};
    ConcatNeighboursBand<<<blocks_mean, threads_mean>>>(old_img, new_img);

    cudaFree(old_img.data);

    return new_img;
}

__global__ void MulImages(Matrix img, std::size_t* position, std::size_t pos_size, Matrix output)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < pos_size && y < img.bands_height)
    {
        auto pixel_pos = position[x];
        auto value = GetElement(img, y, pixel_pos);
        SetElement(output, y, x, value);
    }
}

std::vector<std::size_t> PositionFromMask(Matrix mask)
{
    assert(mask.pixels_width > 0);
    assert(mask.bands_height == 1);

    std::vector<std::size_t> position;
    for (std::size_t i = 0; i < mask.pixels_width; ++i)
    {
        if (mask.data[i] != 0)
        {
            position.push_back(i);
        }
    }
    return position;
}

CpuMatrix GetObjectFromMask(Matrix img, Matrix mask)
{
    assert(img.pixels_width == mask.pixels_width);

    const std::vector<std::size_t> position = PositionFromMask(mask);
    const std::size_t pixels = position.size();

    Matrix new_img{img.bands_height, pixels, nullptr};
    std::size_t *m_pos = nullptr;
    Matrix old_img{img.bands_height, img.pixels_width, nullptr};

    CudaAssert(cudaMalloc(&old_img.data, old_img.bands_height * old_img.pixels_width * sizeof(float)));
    CudaAssert(cudaMalloc(&m_pos, pixels * sizeof(std::size_t)));
    CudaAssert(cudaMalloc(&new_img.data, new_img.bands_height * new_img.pixels_width * sizeof(float)));

    CudaAssert(cudaMemcpy(old_img.data, img.data, old_img.bands_height * old_img.pixels_width * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemcpy(m_pos, position.data(), pixels * sizeof(std::size_t), cudaMemcpyHostToDevice));

    dim3 threads_mean{32, 32};
    dim3 blocks_mean{static_cast<unsigned int>(pixels) / 32 + 1, static_cast<unsigned int>(old_img.bands_height / 32 + 1)};
    MulImages<<<blocks_mean, threads_mean>>>(old_img, m_pos, pixels, new_img);

    std::shared_ptr<float[]> cpu_ptr = std::make_shared<float[]>(new_img.bands_height * pixels);

    CudaAssert(cudaMemcpy(cpu_ptr.get(), new_img.data, new_img.bands_height * pixels * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(old_img.data);
    cudaFree(m_pos);
    cudaFree(new_img.data);

    ImageSize size{1, static_cast<uint32_t>(new_img.pixels_width), static_cast<uint32_t>(new_img.bands_height)};

    return {size, std::move(cpu_ptr)};
}

__global__ void MatMul(const Matrix a, const Matrix b, const Matrix c)
{
    assert(a.pixels_width == b.bands_height);

    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= c.pixels_width || y >= c.bands_height)
        return;

    float value = 0.f;
    for (std::size_t i = 0; i < a.pixels_width; ++i)
    {
        value += GetElement(a, y, i) * GetElement(b, i, x);
    }
    AddElement(c, y, x, value);
}


std::vector<CpuMatrix> MatmulPcaEigenvectors(CpuMatrix &eigenvectors, std::size_t k_bands,
    std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_pixels, std::size_t data_count)
{
    // Matmul [k_bands, bands] x [bands, pixels]

    auto blocking_load_img = [&, max_pixels](std::size_t i, Matrix &img) -> ImageSize {
        auto [size, ptr] = LoadData(i);

        img.pixels_width = size.width * size.height;
        img.bands_height = size.depth;

        assert(img.pixels_width <= max_pixels);

        CudaAssert(cudaMemcpy(img.data, ptr.get(), size.height * size.width * size.depth * sizeof(float), cudaMemcpyHostToDevice));
        return size;
    };

    auto GetCpuMatrix = [k_bands](Matrix img, ImageSize size) -> CpuMatrix {
        ImageSize cpu_size = {.width = size.width, .height = size.height, .depth = static_cast<uint32_t>(k_bands)};

        std::shared_ptr<float[]> cpu_ptr = std::make_shared<float[]>(k_bands * size.width * size.height);
        CudaAssert(cudaMemcpy(cpu_ptr.get(), img.data, k_bands * size.width * size.height * sizeof(float), cudaMemcpyDeviceToHost));

        return CpuMatrix{cpu_size, std::move(cpu_ptr)};
    };

    const auto bands = eigenvectors.size.width;

    assert(data_count >= 1);
    assert(k_bands < bands);

    Matrix c_eigenvectors{k_bands, bands, nullptr};
    Matrix c_img{bands, max_pixels, nullptr};
    Matrix c_img_to_copy{bands, max_pixels, nullptr};
    Matrix c_result{k_bands, max_pixels, nullptr};

    CudaAssert(cudaMalloc(&c_eigenvectors.data, k_bands * bands * sizeof(float)));
    CudaAssert(cudaMalloc(&c_img.data, bands * max_pixels * sizeof(float)));
    CudaAssert(cudaMalloc(&c_img_to_copy.data, bands * max_pixels * sizeof(float)));
    CudaAssert(cudaMalloc(&c_result.data, k_bands * max_pixels * sizeof(float)));

    CudaAssert(cudaMemcpy(c_eigenvectors.data, eigenvectors.data.get() + (bands - k_bands) * bands, k_bands * bands * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(c_result.data, 0.f, k_bands * max_pixels * sizeof(float)));

    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    dim3 threads{32, 32};
    dim3 blocks{static_cast<unsigned int>(max_pixels) / 32 + 1, static_cast<unsigned int>(k_bands / 32 + 1)};

    std::vector<CpuMatrix> results;

    ImageSize loaded_img_size = blocking_load_img(0, c_img);

    for (std::size_t i = 0; i < data_count - 1; ++i)
    {
        MatMul<<<blocks, threads, 0, stream1>>>(c_eigenvectors, c_img, c_result);

        ImageSize loaded_img_size2 = blocking_load_img(i + 1, c_img_to_copy);

        // wait for stream
        cudaStreamSynchronize(stream1);
        results.push_back(GetCpuMatrix(c_result, loaded_img_size));

        std::swap(c_img, c_img_to_copy);
        std::swap(loaded_img_size, loaded_img_size2);
    }
    MatMul<<<blocks, threads, 0, stream1>>>(c_eigenvectors, c_img, c_result);
    cudaStreamSynchronize(stream1);
    results.push_back(GetCpuMatrix(c_result, loaded_img_size));

    cudaStreamDestroy(stream1);
    cudaFree(c_eigenvectors.data);
    cudaFree(c_img.data);
    cudaFree(c_img_to_copy.data);
    cudaFree(c_result.data);

    return results;
}


__global__ void CalculateFourMovements(Matrix img, Matrix result)
{
    const auto y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < img.bands_height)
    {
        float sum2 = 0;
        float sum3 = 0;
        float sum4 = 0;

        for (std::size_t x = 0; x < img.pixels_width; ++x)
        {
            const auto pixel = GetElement(img, y, x);

            const float val2 = pixel * pixel;
            const float val3 = val2 * pixel;
            const float val4 = val3 * pixel;

            sum2 += val2;
            sum3 += val3;
            sum4 += val4;
        }

        sum2 /= static_cast<float>(img.pixels_width);
        sum3 /= static_cast<float>(img.pixels_width);
        sum4 /= static_cast<float>(img.pixels_width);

        SetElement(result, y, 0, sum2);
        SetElement(result, y, 1, sum3);
        SetElement(result, y, 2, sum4);
    }
}

std::vector<StatisticalParameters> GetStatistics(CpuMatrix cpu_img)
{
    assert(cpu_img.data != nullptr);

    Matrix img{cpu_img.size.depth, cpu_img.size.width * cpu_img.size.height, nullptr};
    Matrix mean{img.bands_height, 1, nullptr};

    Matrix four_movements{img.bands_height, 3, nullptr};

    CudaAssert(cudaMalloc(&img.data, img.bands_height * img.pixels_width * sizeof(float)));
    CudaAssert(cudaMalloc(&mean.data, mean.bands_height * mean.pixels_width * sizeof(float)));
    CudaAssert(cudaMalloc(&four_movements.data, four_movements.bands_height * four_movements.pixels_width * sizeof(float)));

    CudaAssert(cudaMemcpy(img.data, cpu_img.data.get(), img.bands_height * img.pixels_width * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(mean.data, 0, mean.bands_height * mean.pixels_width * sizeof(float)));

    dim3 threads_sum{32, 32};
    dim3 blocks_sum{static_cast<unsigned int>(img.pixels_width / 32 + 1), static_cast<unsigned int>(img.bands_height / 32 + 1)};

    dim3 threads_division{1, 1024};
    dim3 blocks_division{1, static_cast<unsigned int>(mean.bands_height/ 32 + 1)};

    dim3 threads_subtract{32, 32};
    dim3 blocks_subtract{static_cast<unsigned int>(img.pixels_width / 32 + 1), static_cast<unsigned int>(img.bands_height / 32 + 1)};

    dim3 threads_movement{1024};
    dim3 blocks_movement{static_cast<unsigned int>(img.bands_height/ 32 + 1)};


    /// START CUDA PIPELINE
    SumRows<<<blocks_sum, threads_sum>>>(img, mean);

    PieceWiseDivision<<<blocks_division, threads_division>>>(mean, static_cast<float>(img.pixels_width));

    SubtractMean<<<blocks_subtract, threads_subtract>>>(img, mean);

    CalculateFourMovements<<<blocks_movement, threads_movement>>>(img, four_movements);

    cudaDeviceSynchronize();
    /// END CUDA PIPELINE


    std::unique_ptr<float[]> cpu_mean{new float[img.bands_height]};
    std::unique_ptr<float[]> cpu_movements{new float[img.bands_height * 3]};

    CudaAssert(cudaMemcpy(cpu_mean.get(), mean.data, img.bands_height *  sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaMemcpy(cpu_movements.get(), four_movements.data, img.bands_height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<StatisticalParameters> result;
    for (std::size_t i = 0; i < img.bands_height; ++i)
    {
        const std::size_t idx = i * 3;

        const float mean_value = cpu_mean[i];
        const float second_movement = cpu_movements[idx]; // variance
        const float third_movement = cpu_movements[idx + 1];
        const float fourth_movement = cpu_movements[idx + 2];

        const float std_dev = sqrt(second_movement);

        const float skewness = third_movement / std::pow(std_dev, 3);
        const float kurtosis = fourth_movement / std::pow(std_dev, 4);

        result.push_back(StatisticalParameters{mean_value, second_movement, skewness, kurtosis});
    }

    cudaFree(img.data);
    cudaFree(mean.data);
    cudaFree(four_movements.data);

    return result;
}
