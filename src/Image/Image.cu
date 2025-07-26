#include "Image.hpp"

// #include "Classification.hpp"
#include "EntityComponentSystem.hpp"

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <map>
#include <numeric>
#include <span>
#include <cmath>
#include <fstream>


/// Stupid winapi macro name
#undef LoadImage

extern Coordinator coordinator;



Entity CreateImage(const FilesystemPaths &paths)
{
    const auto id = coordinator.CreateEntity();

    coordinator.AddComponent(id, paths);

    const auto opt_envi = LoadEnvi(paths.envi_path);
    if (!opt_envi.has_value())
    {
        const auto file = paths.envi_path.string();
        LOG_ERROR("CreateImage(): failed to load ENVI file {}!", file);
        throw std::runtime_error{"Empty envi header"};
    }
    coordinator.AddComponent(id, opt_envi.value());
    coordinator.AddComponent(
        id,
        ImageSize{
            opt_envi->samples_per_image,
            opt_envi->lines_per_image,
            opt_envi->bands_number}
        );

    LOG_INFO("Created image id={}", id);
    return id;
}


std::shared_ptr<float[]> LoadImage(std::istream &iss, const EnviHeader &envi)
{
    assert(envi.byte_order == ByteOrder::LITTLE_ENDIAN);

    std::shared_ptr<float[]> host_data{new float[envi.bands_number *
                                       envi.lines_per_image *
                                       envi.samples_per_image]};


    using AccessScheme = std::function<float*(std::size_t, std::size_t, std::size_t)>;
    AccessScheme access_scheme;

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


    auto LoaderFn = [&]<typename T>() -> void {
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
    };

    switch (envi.data_type)
    {
        case DataType::BYTE:
            LoaderFn.operator()<char>();
            break;
        case DataType::INT16:
            LoaderFn.operator()<int16_t>();
            break;
        case DataType::INT32:
            LoaderFn.operator()<int32_t>();
            break;
        case DataType::INT64:
            LoaderFn.operator()<int64_t>();
            break;
        case DataType::UINT16:
            LoaderFn.operator()<uint16_t>();
            break;
        case DataType::UINT32:
            LoaderFn.operator()<uint32_t>();
            break;
        case DataType::UINT64:
            LoaderFn.operator()<uint64_t>();
            break;
        case DataType::FLOAT32:
            LoaderFn.operator()<float>();
            break;
        case DataType::FLOAT64:
            LoaderFn.operator()<double>();
            break;
        case DataType::COMPLEX32:
        case DataType::COMPLEX64:
            LOG_ERROR("LoadImage unsupported data type: {}", static_cast<int>(envi.data_type));
            return nullptr;
    }
    return host_data;
}

CpuMatrix GetImageData(const Entity entity)
{
    static std::map<Entity, std::weak_ptr<float[]>> loaded_img{};

    const auto &size = coordinator.GetComponent<ImageSize>(entity);

    const auto iter = loaded_img.find(entity);
    if (iter != loaded_img.end() && !iter->second.expired())
    {
        return CpuMatrix{size, iter->second.lock()};
    }

    const auto &path = coordinator.GetComponent<FilesystemPaths>(entity).img_path;
    const auto &envi = coordinator.GetComponent<EnviHeader>(entity);

    std::ifstream file{path, std::ios_base::binary | std::ios::in};
    assert(file.is_open());

    std::shared_ptr<float[]> ptr = LoadImage(file, envi);
    loaded_img[entity] = ptr;

    return CpuMatrix{size, std::move(ptr)};
}


__global__ void Mean(GpuMatrix img, GpuMatrix mean)
{
    assert(mean.size.channel == img.size.channel);
    assert(mean.size.width == 1);
    assert(mean.size.height == 1);

    const auto ch = blockIdx.x * blockDim.x + threadIdx.x;

    const auto image_resolution = img.size.width * img.size.height;

    if (ch < img.size.channel)
    {
        float result = 0.0f;

        for (std::size_t pixel = 0; pixel < image_resolution; ++pixel)
        {
            result += img.get(ch, pixel);
        }
        result /= static_cast<float>(image_resolution);
        img.get(ch, 0) = result;
    }
}

__global__ void SumRows(GpuMatrix img, GpuMatrix sum)
{
    const auto ch = blockIdx.x * blockDim.x + threadIdx.x;
    const auto image_resolution = img.size.width * img.size.height;

    if (ch < img.size.channel)
    {
        float result = 0.0f;

        for (std::size_t pixel = 0; pixel < image_resolution; ++pixel)
        {
            result += img.get(ch, pixel);
        }
        img.get(ch, 0) += result;
    }
}

__global__ void PieceWiseDivision(GpuMatrix m, float divisor)
{
    const auto pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const auto ch = blockIdx.y * blockDim.y + threadIdx.y;

    const auto image_resolution = m.size.width * m.size.height;

    if (pixel < image_resolution && ch < m.size.channel)
    {
        const float result = m.get(ch, pixel) / divisor;
        m.get(ch, pixel) = result;
    }
}

__global__ void SubtractMean(GpuMatrix img, GpuMatrix mean)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto ch = blockIdx.y * blockDim.y + threadIdx.y;

    const auto image_resolution = img.size.width * img.size.height;

    if (ch < img.size.channel && x < image_resolution)
    {
        const float value = img.get(ch, x) - mean.get(ch,  0);
        img.get(ch, x) = value;
    }
}


__global__ void MatMulTrans(const GpuMatrix img, const GpuMatrix result)
{
    const std::size_t ch = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    const auto image_resolution = img.size.width * img.size.height;

    if (x >= image_resolution || ch >= result.size.channel)
        return;

    float value = 0.f;
    for (std::size_t i = 0; i < image_resolution; ++i)
    {
        // X * X^T
        value += img.get(ch, i) * img.get(x, i);
    }
    result.get(ch, x) += value;
}

GpuMatrix CovarianceMatrix(std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_height, uint32_t max_width, std::size_t data_count)
{
    // pixels_width = x = pixels_width = ImageSize.max_width * ImageSize.height, bands_height = y = bands_height= ImageSize.channel

    auto blocking_load_img = [&, max_height, max_width](std::size_t i, GpuMatrix &img) -> ImageSize {
        const auto [size, cpu_ptr] = LoadData(i);
        img.size = size;
        CudaAssert(cudaMemcpy(img.data, cpu_ptr.get(), size.height * size.width * size.channel * sizeof(float), cudaMemcpyHostToDevice));
        return size;
    };

    GpuMatrix img{{0, 0, 0},  nullptr};
    GpuMatrix mean{{0 , 1, max_height}, nullptr};
    GpuMatrix cov{{max_height, max_height, 1}, nullptr};

    GpuMatrix img_to_copy{{0, 0, 0},  nullptr};

    CudaAssert(cudaMallocHost(&img.data, max_height * max_width * sizeof(float)));
    CudaAssert(cudaMallocHost(&img_to_copy.data, max_height * max_width * sizeof(float)));

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

    LOG_INFO("Start calculation covariance matrix");

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
    PieceWiseDivision<<<blocks_division, threads_division, 0, stream1>>>(mean, static_cast<float>(img.size.width * img.size.height * data_count));
    CudaAssert(cudaStreamSynchronize(stream1));

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
    CudaAssert(cudaStreamSynchronize(stream1));

    PieceWiseDivision<<<blocks_division_2, threads_division_2, 0, stream1>>>(cov, static_cast<float>(img.size.width * img.size.height * data_count));
    CudaAssert(cudaStreamSynchronize(stream1));

    LOG_INFO("End covariance matrix");

    CudaAssert(cudaFreeHost(img.data));
    CudaAssert(cudaFreeHost(img_to_copy.data));
    CudaAssert(cudaFree(mean.data));

    CudaAssert(cudaStreamDestroy(stream1));

    return cov;
}

ResultPCA PCA(std::function<CpuMatrix(std::size_t)> LoadData, uint32_t max_height, uint32_t max_width, std::size_t data_count)
{
    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    GpuMatrix cov = CovarianceMatrix(LoadData, max_height, max_width, data_count);

    float *d_eigenvalues = nullptr;
    CudaAssert(cudaMalloc(&d_eigenvalues, max_height * sizeof(float)));

    // Calculate eigenvalues
    cusolverDnHandle_t handle = nullptr;
    int *dev_info = nullptr;
    int lwork = 0; // size of workspace
    float *d_work = nullptr;
    constexpr cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    const int size = static_cast<int>(cov.size.channel);

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


    auto eigenvector = std::make_shared<float[]>(cov.size.channel * cov.size.width * cov.size.height);
    auto eigenvalues = std::make_shared<float[]>(cov.size.width * cov.size.height);

    cudaMemcpy(eigenvector.get(), cov.data, cov.size.channel * cov.size.width * cov.size.height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvalues.get(), d_eigenvalues, cov.size.width * cov.size.height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    CudaAssert(cudaFree(d_eigenvalues));

    CusolverAssert(cusolverDnDestroy(handle));
    CudaAssert(cudaStreamDestroy(stream1));

    CudaAssert(cudaDeviceReset());

    CpuMatrix mat_eigenvalues{
        .size = ImageSize{
            .width = 1,
            .height = static_cast<uint32_t>(cov.size.channel),
            .channel = 1},
        .data = std::move(eigenvalues)
    };
    CpuMatrix mat_eigenvectors{
        .size = ImageSize{
            .width = static_cast<uint32_t>(cov.size.width),
            .height = static_cast<uint32_t>(cov.size.height),
            .channel = 1},
        .data = std::move(eigenvector)
    };


    return {.eigenvalues = mat_eigenvalues, .eigenvectors = mat_eigenvectors};
}

__global__ void Threshold(GpuMatrix img, std::size_t band, float threshold, float *mask)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < img.size.width * img.size.height)
    {
        const float value = img.get(band, 0, x) > threshold ? 1.f : 0.f;
        mask[x] = value;
    }
}

CpuMatrix ManualThresholding(CpuMatrix img, std::size_t band, float threshold)
{
    GpuMatrix d_img{{.width = img.size.width, .height = img.size.height, .channel = 1}, nullptr};
    float *d_mask = nullptr;

    const auto img_resolution = img.size.width * img.size.height;

    CudaAssert(cudaMalloc(&d_img.data, d_img.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&d_mask, img_resolution * sizeof(float)));

    const auto* ptr_offset = img.data.get() + img_resolution * band;
    CudaAssert(cudaMemcpy(d_img.data, ptr_offset, d_img.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(d_mask, 0, img_resolution * sizeof(float)));

    dim3 threads_mean{1024};
    dim3 blocks_mean{static_cast<unsigned int>(img_resolution) / 1024 + 1};
    Threshold<<<blocks_mean, threads_mean>>>(d_img, band, threshold, d_mask);
    CudaAssert(cudaDeviceSynchronize());

    std::shared_ptr<float[]> mask{new float[img_resolution]};

    CudaAssert(cudaMemcpy(mask.get(), d_mask, img_resolution * sizeof(float), cudaMemcpyDeviceToHost));

    CudaAssert(cudaFree(d_img.data));
    CudaAssert(cudaFree(d_mask));

    return {d_img.size, std::move(mask)};
}

__global__ void ConcatNeighboursBand(GpuMatrix old_img, GpuMatrix new_img)
{
    static constexpr std::size_t up_left_offset     =  1;
    static constexpr std::size_t up_center_offset   =  2;
    static constexpr std::size_t up_right_offset    =  3;
    static constexpr std::size_t mid_left_offset    =  4;
    static constexpr std::size_t mid_right_offset   =  5;
    static constexpr std::size_t down_left_offset   =  6;
    static constexpr std::size_t down_center_offset =  7;
    static constexpr std::size_t down_right_offset  =  8;

    static constexpr int max_x_threads = 1024;
    static constexpr int block_height = 3;
    static constexpr int max_block_width = max_x_threads;

    const auto block_start = blockIdx.x * (blockDim.x - 2);

    const auto x = block_start + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    const auto ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= old_img.size.width || y >= new_img.size.height || ch >= old_img.size.channel)
        return;

    const auto old_i = (y + 1) * old_img.size.width + x;
    const auto old_up_i = old_i - old_img.size.width;
    const auto old_down_i = old_i + old_img.size.width;

    const auto block_width = (block_start + max_x_threads < old_img.size.width) ? max_block_width : (old_img.size.width - block_start);

    __shared__ float temp[max_block_width * block_height];

    // Fill upper row
    temp[threadIdx.x] = old_img.get(ch, old_up_i);

    // Fill center
    temp[block_width + threadIdx.x] = old_img.get(ch, old_i);

    // Fill lower row
    temp[block_width * 2 + threadIdx.x] = old_img.get(ch, old_down_i);

    __syncthreads();

    if (threadIdx.x == 0 || threadIdx.x == block_width - 1)
        return;

    // for the second line in the same with
    const auto temp_i = threadIdx.x; // value in range [1, block_width-2]

    const auto up_left =   temp[temp_i - 1];
    const auto up_center = temp[temp_i];
    const auto up_right =  temp[temp_i + 1];

    const auto mid_left =   temp[block_width + temp_i - 1];
    const auto mid_center = temp[block_width + temp_i];
    const auto mid_right =  temp[block_width + temp_i + 1];

    const auto down_left =   temp[block_width * 2 + temp_i - 1];
    const auto down_center = temp[block_width * 2 + temp_i];
    const auto down_right =  temp[block_width * 2 + temp_i + 1];

    const int band_offset = old_img.size.channel;

    const auto i = y * new_img.size.width + x - 1;
    new_img.get(ch, i) = mid_center;


    // Neighbours bands
    new_img.get(ch + band_offset * up_left_offset,   i) = up_left;
    new_img.get(ch + band_offset * up_center_offset, i) = up_center;
    new_img.get(ch + band_offset * up_right_offset,  i) = up_right;

    new_img.get(ch + band_offset * mid_left_offset,  i) = mid_left;
    new_img.get(ch + band_offset * mid_right_offset, i) = mid_right;

    new_img.get(ch + band_offset * down_left_offset,   i) = down_left;
    new_img.get(ch + band_offset * down_center_offset, i) = down_center;
    new_img.get(ch + band_offset * down_right_offset,  i) = down_right;
}

CpuMatrix AddNeighboursBand(CpuMatrix img)
{
    const ImageSize new_size{
        .width = img.size.width - 2,
        .height = img.size.height - 2,
        .channel = img.size.channel * 9
    };

    GpuMatrix old_img{img.size, nullptr};
    GpuMatrix new_img{new_size, nullptr};


    CudaAssert(cudaMalloc(&old_img.data, old_img.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&new_img.data, new_img.elements() * sizeof(float)));

    CudaAssert(cudaMemcpy(old_img.data, img.data.get(), old_img.elements() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads{1024, 1, 1};
    dim3 blocks{
        static_cast<unsigned int>(img.size.width / 1024 + 1),
        static_cast<unsigned int>(new_size.height),
        static_cast<unsigned int>(old_img.size.height * old_img.size.width)
    };
    ConcatNeighboursBand<<<blocks, threads>>>(old_img, new_img);
    cudaFree(old_img.data);

    CpuMatrix cpu_matrix{
        new_size,
        std::shared_ptr<float[]>(new float[new_size.width * new_size.height * new_size.channel])
    };

    CudaAssert(cudaMemcpy(cpu_matrix.data.get(), new_img.data, sizeof(float) * new_img.elements(), cudaMemcpyDeviceToHost));

    cudaFree(new_img.data);

    return std::move(cpu_matrix);
}

__global__ void MulImages(GpuMatrix img, std::size_t* position, std::size_t pos_size, GpuMatrix output)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto ch = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < pos_size && ch < img.size.channel)
    {
        const auto pixel_pos = position[x];
        const auto value = img.get(ch, pixel_pos);
        output.get(ch, x) = value;
    }
}

CpuMatrix GetObjectFromMask(CpuMatrix img, CpuMatrix mask)
{
    assert(img.size.width == mask.size.width);
    assert(img.size.height == mask.size.height);

    const std::vector<std::size_t> position = [&mask]() -> std::vector<std::size_t> {
        assert(mask.size.width > 0);
        assert(mask.size.height > 0);
        assert(mask.size.channel == 1);

        std::vector<std::size_t> position;
        for (std::size_t i = 0; i < mask.size.width * mask.size.height; ++i)
        {
            if (mask.data[i] != 0)
            {
                position.push_back(i);
            }
        }
        return position;
    }();

    const std::size_t pixels = position.size();

    GpuMatrix new_img{img.size, nullptr};
    std::size_t *m_pos = nullptr;
    GpuMatrix old_img{img.size, nullptr};

    CudaAssert(cudaMalloc(&old_img.data, old_img.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&m_pos, pixels * sizeof(std::size_t)));
    CudaAssert(cudaMalloc(&new_img.data, new_img.elements() * sizeof(float)));

    CudaAssert(cudaMemcpy(old_img.data, img.data.get(), old_img.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemcpy(m_pos, position.data(), pixels * sizeof(std::size_t), cudaMemcpyHostToDevice));

    dim3 threads_mean{32, 32};
    dim3 blocks_mean{static_cast<unsigned int>(pixels) / 32 + 1, static_cast<unsigned int>(old_img.size.channel / 32 + 1)};
    MulImages<<<blocks_mean, threads_mean>>>(old_img, m_pos, pixels, new_img);

    std::shared_ptr<float[]> cpu_ptr = std::make_shared<float[]>(new_img.size.channel * pixels);

    CudaAssert(cudaMemcpy(cpu_ptr.get(), new_img.data, new_img.size.channel * pixels * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(old_img.data);
    cudaFree(m_pos);
    cudaFree(new_img.data);

    return {new_img.size, std::move(cpu_ptr)};
}

__global__ void MatMul(const GpuMatrix a, const GpuMatrix b, const GpuMatrix c)
{
    assert(a.size.width * a.size.height == b.size.channel);
    assert(a.size.channel == c.size.channel);

    assert(b.size.width == c.size.width);
    assert(b.size.height == c.size.height);

    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t ch = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= c.size.height * c.size.width || ch >= c.size.channel)
        return;

    float value = 0.f;
    for (std::size_t i = 0; i < a.size.width * a.size.height; ++i)
    {
        value += a.get(ch, i) * b.get(i, x);
    }
    c.get(ch, x) = value;
}


// TODO: simplify
std::vector<CpuMatrix> MatmulPcaEigenvectors(const CpuMatrix &eigenvectors, ImageSize new_size,
    std::function<CpuMatrix(std::size_t)> LoadData, std::size_t data_count)
{
    // Matmul [k_bands, bands] x [bands, pixels]

    const auto bands = eigenvectors.size.width;
    assert(data_count >= 1);
    assert(new_size.channel < bands);

    auto blocking_load_img = [&, new_size](std::size_t i, GpuMatrix &img) {
        const auto [size, ptr] = LoadData(i);

        assert(size.channel == bands);

        img.size = size;
        CudaAssert(cudaMemcpy(img.data, ptr.get(), img.elements() * sizeof(float), cudaMemcpyHostToDevice));
    };

    auto GetCpuMatrix = [new_size](GpuMatrix img) -> CpuMatrix {
        ImageSize cpu_size = {.width = img.size.width, .height = img.size.height, .channel = static_cast<uint32_t>(new_size.channel)};

        const auto elements = cpu_size.width * cpu_size.height * cpu_size.channel;
        std::shared_ptr<float[]> cpu_ptr = std::make_shared<float[]>(elements);

        CudaAssert(cudaMemcpy(cpu_ptr.get(), img.data, elements * sizeof(float), cudaMemcpyDeviceToHost));

        return CpuMatrix{cpu_size, std::move(cpu_ptr)};
    };

    // TODO: add paged lock mallco to speed up copy
    GpuMatrix c_eigenvectors{{.width = 1, .height = bands, .channel = new_size.channel}, nullptr};
    GpuMatrix c_img{{.width = new_size.width, .height = new_size.height, .channel = bands}, nullptr};
    GpuMatrix c_img_to_copy{{.width = new_size.width, .height = new_size.height, .channel = bands}, nullptr};
    GpuMatrix c_result{new_size, nullptr};

    CudaAssert(cudaMalloc(&c_eigenvectors.data, c_eigenvectors.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&c_img.data, c_img.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&c_img_to_copy.data, c_img_to_copy.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&c_result.data, c_result.elements() * sizeof(float)));

    CudaAssert(cudaMemcpy(c_eigenvectors.data, eigenvectors.data.get(), c_eigenvectors.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(c_result.data, 0.f, c_result.elements() * sizeof(float)));

    cudaStream_t stream1;
    CudaAssert(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    dim3 threads{32, 32};
    dim3 blocks{static_cast<unsigned int>(new_size.width * new_size.height) / 32 + 1,
                static_cast<unsigned int>(new_size.channel/ 32 + 1)};

    std::vector<CpuMatrix> results;

    blocking_load_img(0, c_img);
    assert(c_img.size == new_size);
    // c_result.pixels_width = loaded_img_size.width * loaded_img_size.height; // why ? should be in our case constant size

    for (std::size_t i = 0; i < data_count - 1; ++i)
    {
        MatMul<<<blocks, threads, 0, stream1>>>(c_eigenvectors, c_img, c_result);

        blocking_load_img(i + 1, c_img_to_copy);
        assert(c_img_to_copy.size == new_size);

        // wait for stream
        cudaStreamSynchronize(stream1);
        results.push_back(GetCpuMatrix(c_result));

        std::swap(c_img, c_img_to_copy);

    }
    MatMul<<<blocks, threads, 0, stream1>>>(c_eigenvectors, c_img, c_result);
    cudaStreamSynchronize(stream1);
    results.push_back(GetCpuMatrix(c_result));

    cudaStreamDestroy(stream1);
    cudaFree(c_eigenvectors.data);
    cudaFree(c_img.data);
    cudaFree(c_img_to_copy.data);
    cudaFree(c_result.data);

    return results;
}


__global__ void CalculateFourMovements(GpuMatrix img, GpuMatrix result)
{
    const auto ch = blockIdx.x * blockDim.x + threadIdx.x;
    const auto image_resolution = img.size.width * img.size.height;

    if (ch < img.size.channel)
    {
        float sum2 = 0;
        float sum3 = 0;
        float sum4 = 0;

        for (std::size_t x = 0; x < img.size.width * img.size.height; ++x)
        {
            const auto pixel = img.get(ch, x);

            const float val2 = pixel * pixel;
            const float val3 = val2 * pixel;
            const float val4 = val3 * pixel;

            sum2 += val2;
            sum3 += val3;
            sum4 += val4;
        }

        sum2 /= static_cast<float>(image_resolution);
        sum3 /= static_cast<float>(image_resolution);
        sum4 /= static_cast<float>(image_resolution);

        result.get(ch, 0) = sum2;
        result.get(ch, 1) = sum3;
        result.get(ch, 2) = sum4;
    }
}

std::vector<StatisticalParameters> GetStatistics(const CpuMatrix& cpu_img)
{
    assert(cpu_img.data != nullptr);

    GpuMatrix img{cpu_img.size, nullptr};
    GpuMatrix mean{{.width =  1, .height = 1, .channel = cpu_img.size.channel}, nullptr};

    GpuMatrix four_movements{{3, 1, cpu_img.size.channel}, nullptr};

    CudaAssert(cudaMalloc(&img.data, img.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&mean.data, mean.elements() * sizeof(float)));
    CudaAssert(cudaMalloc(&four_movements.data, four_movements.elements() * sizeof(float)));

    CudaAssert(cudaMemcpy(img.data, cpu_img.data.get(), img.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CudaAssert(cudaMemset(mean.data, 0, mean.elements() * sizeof(float)));

    dim3 threads_sum{1024};
    dim3 blocks_sum{static_cast<unsigned int>(img.size.channel / 32 + 1)};

    dim3 threads_division{1, 1024};
    dim3 blocks_division{1, static_cast<unsigned int>(mean.size.channel / 32 + 1)};

    dim3 threads_subtract{32, 32};
    dim3 blocks_subtract{static_cast<unsigned int>(img.size.width * img.size.height / 32 + 1), static_cast<unsigned int>(img.size.channel / 32 + 1)};

    dim3 threads_movement{1024};
    dim3 blocks_movement{static_cast<unsigned int>(img.size.channel / 32 + 1)};


    /// START CUDA PIPELINE
    SumRows<<<blocks_sum, threads_sum>>>(img, mean);

    PieceWiseDivision<<<blocks_division, threads_division>>>(mean, static_cast<float>(img.size.width * img.size.height));

    SubtractMean<<<blocks_subtract, threads_subtract>>>(img, mean);

    CalculateFourMovements<<<blocks_movement, threads_movement>>>(img, four_movements);

    cudaDeviceSynchronize();
    /// END CUDA PIPELINE


    std::unique_ptr<float[]> cpu_mean{new float[img.size.channel]};
    std::unique_ptr<float[]> cpu_movements{new float[img.size.channel * 3]};

    CudaAssert(cudaMemcpy(cpu_mean.get(), mean.data, img.size.channel *  sizeof(float), cudaMemcpyDeviceToHost));
    CudaAssert(cudaMemcpy(cpu_movements.get(), four_movements.data, img.size.channel * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<StatisticalParameters> result;
    for (std::size_t i = 0; i < img.size.channel; ++i)
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

ImageLabel::ImageLabel(const std::filesystem::path &file_path, const ImageSize size): img_size_{size}
{
    assert(!file_path.empty());
    assert(size.width > 0);
    assert(size.height > 0);

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        LOG_ERROR("ImageLabel: Could not open file {}", file_path.string());
        throw std::runtime_error("Could not open file");
    }

    LOG_INFO("Loading {} labels", img_size_.width * img_size_.height);
    image_label_.resize(img_size_.width * img_size_.height);
    for (std::size_t i = 0; i < img_size_.width * img_size_.height; ++i)
    {
        file >> image_label_[i];
    }
}

uint8_t ImageLabel::GetLabels(const PatchData patch_pos)
{
    const auto [x, y] = patch_pos;
    const auto idx = y * img_size_.width + x;
    return image_label_.at(idx);
}

CpuMatrix GetImportantEigenvectors(const CpuMatrix &eigenvectors, std::size_t k_bands)
{
    const auto [width, height, band] = eigenvectors.size;

    const int ptr_diff = (static_cast<int>(height) - static_cast<int>(k_bands)) * static_cast<int>(width);

    std::shared_ptr<float[]> eigenvectors_data(new float[k_bands * width]);
    memcpy(eigenvectors_data.get(), eigenvectors.data.get() + ptr_diff, k_bands * width * sizeof(float));

    ImageSize size{width, static_cast<uint32_t>(k_bands), 1};
    return CpuMatrix{size, std::move(eigenvectors_data)};
}

float SumAllCuda(CpuMatrix data)
{
    float *c_ptr = nullptr;
    CudaAssert(cudaMalloc(&c_ptr, data.elements() * sizeof(float)));
    CudaAssert(cudaMemcpy(c_ptr, data.data.get(), data.elements() * sizeof(float), cudaMemcpyHostToDevice));

    thrust::device_vector<float> c_vec(c_ptr, c_ptr + data.elements());
    return thrust::reduce(c_vec.begin(), c_vec.end());
}

PatchSystem::PatchSystem(Entity parent_img): parent_img{parent_img}
{
    const auto [size, img_data] = GetImageData(parent_img);
    size_ = size;
    img_data_ = img_data;
}

std::size_t PatchSystem::GetPatchNumbers(ImageSize size)
{
    return size.width * size.height;
}

CpuMatrix PatchSystem::GetPatchImage(int center_x, int center_y) const
{
    static constexpr int margin = PatchData::S / 2;

    const std::size_t band_offset = size_.width * size_.height;
    const std::size_t height_offset = size_.width;

    CpuMatrix result{
        ImageSize{S, S, size_.channel},
        std::make_shared<float[]>(S * S * size_.channel)
    };

    for (int band = 0; band < size_.channel; band++)
    {
        for (int y = center_y - margin, iy = 0; y < center_y + margin; ++y, ++iy)
        {
            for (int x = center_x - margin, ix = 0; x < center_x + margin; ++x, ++ix)
            {
                float *value = result.data.get() + iy * S + ix + band * S * S;

                if (x < 0 || x >= size_.width || y < 0 || y >= size_.height)
                    *value = 0;
                else
                    *value = img_data_[band * band_offset + y * height_offset + x];
            }
        }
    }

    return std::move(result);
}

// float KernelRbfThrust(const AttributeList &a1, const AttributeList &a2, const float gamma)
// {
//     assert(!a1.empty() && !a2.empty());
//     assert(a1.size() == a2.size());
//
//     auto power_2 = []  __host__ __device__ (float x) { return x * x; };
//
//     std::vector<float> difference(a1.size(), 0);
//     thrust::transform(thrust::host, a1.begin(), a1.end(), a2.begin(), difference.begin(), thrust::minus<float>());
//
//     auto begin_iter = thrust::make_transform_iterator(difference.begin(), power_2);
//     auto end_iter = thrust::make_transform_iterator(difference.end(), power_2);
//
//     const float l2_power = thrust::reduce(begin_iter, end_iter, 0.f);
//
//     return std::exp(-gamma * l2_power);
// }


CpuMatrix MultiplyMask(CpuMatrix threshold_mask, CpuMatrix segmentation_mask)
{
    CpuMatrix mask{
        .size = threshold_mask.size,
        .data = std::make_shared<float[]>(threshold_mask.size.width * threshold_mask.size.height)
    };

    for (auto i = 0; i < mask.size.width * mask.size.height; ++i)
    {
        if (threshold_mask.data[i] == 1.f && segmentation_mask.data[i] ==  1.f)
            mask.data[i] = 1.f;
        else
            mask.data[i] = 0.f;
    }
    return std::move(mask);
}

__global__ void CudaSAM(GpuMatrix img, std::size_t i, std::size_t j, float *pi_arr, float *pj_arr, float *pij_arr)
{
    const auto band = blockIdx.x * blockDim.x + threadIdx.x;

    if (band < img.size.channel)
    {
        const float pi = img.data[i + band * img.size.width * img.size.height];
        const float pj = img.data[j + band * img.size.width * img.size.height];

        pi_arr[band] = pi * pi;
        pj_arr[band] = pj * pj;
        pij_arr[band] = pi * pj;
    }
}


CpuMatrix SegmentationSAM(CpuMatrix img, float radian_threshold)
{
    GpuMatrix m_img{img.size, nullptr};

    // if (m_img.bands_height > 1024)
    // {
    //     LOG_ERROR("Too much spectral bands");
    //     throw std::runtime_error("Too much spectral bands");
    // }

    CudaAssert(cudaMalloc(&m_img.data, m_img.elements() * sizeof(float)));
    CudaAssert(cudaMemcpy(m_img.data, img.data.get(), m_img.elements() * sizeof(float), cudaMemcpyHostToDevice));

    float *pi_arr = nullptr;
    float *pj_arr = nullptr;
    float *pij_arr = nullptr;

    CudaAssert(cudaMalloc(&pi_arr, m_img.size.channel * sizeof(float)));
    CudaAssert(cudaMalloc(&pj_arr, m_img.size.channel * sizeof(float)));
    CudaAssert(cudaMalloc(&pij_arr, m_img.size.channel * sizeof(float)));

    // Central pixel is J
    const std::size_t center_x = img.size.width / 2;
    const std::size_t center_y = img.size.height / 2;
    const std::size_t j = center_y * img.size.width + center_x;

    std::vector<float> pixel_sam(m_img.size.width * m_img.size.height, 0.f);

    for (std::size_t i = 0; i < m_img.size.width * m_img.size.height; ++i)
    {
        if (i == j)
            continue;

        CudaSAM<<<(m_img.size.channel / 1024) + 1, 1024>>>(m_img, i, j, pi_arr, pj_arr, pij_arr);
        // cudaDeviceSynchronize();

        const float sum_pi = thrust::reduce(thrust::device, pi_arr, pi_arr + m_img.size.channel , 0.f);
        const float sum_pj = thrust::reduce(thrust::device, pj_arr, pj_arr + m_img.size.channel, 0.f);
        const float sum_pij = thrust::reduce(thrust::device, pij_arr, pij_arr + m_img.size.channel , 0.f);

        const float sam_value = std::acos(sum_pij / std::sqrt(sum_pi * sum_pj));

        pixel_sam[i] = sam_value;
    }
    // LOG_INFO("Radian sam: {}", fmt::join(pixel_sam, ","));

    float *mask_data = new float[m_img.size.width * m_img.size.height];

    CpuMatrix mask{
        .size = ImageSize{.width = img.size.width, .height = img.size.height, .channel = 1},
        .data = std::shared_ptr<float[]>(mask_data)
    };

    for (std::size_t i = 0; i < mask.size.width * mask.size.height; ++i)
    {
        if (i == j)
            mask.data[i] = 1;

        if (pixel_sam[i] <= radian_threshold)
        {
            mask.data[i] = 1;
        }
        else
        {
            mask.data[i] = 0;
        }
    }
    cudaFree(pi_arr);
    cudaFree(pj_arr);
    cudaFree(pij_arr);
    cudaFree(m_img.data);

    return std::move(mask);
}



__device__ float CudaRBF(float *x1, float *x2, std::size_t size)
{
    float sum = 0;
    for (std::size_t i = 0; i < size; ++i)
    {
        sum += powf(x1[i] - x2[i], 2);
    }
    return sum;
}


__global__ void FunctionValueSVM(float *alpha_y, float *x, float *f, float *data, float gamma, std::size_t size, std::size_t obj_size)
{
    const auto y = blockIdx.x * blockDim.x + threadIdx.x;
    const auto i = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO add __shared for alpha, x

    if (i < size && y < obj_size)
    {
        const auto offset = y * size;
        f[i + offset] = alpha_y[i] * CudaRBF(data + i, x + i, gamma);
    }
}


__global__ void GenerateKey(int *key, std::size_t obj_size, std::size_t size)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size * obj_size)
    {
        const int value = ((x / size) % 2 == 0) ? 1 : 0;
        key[x] = value;
    }
}

PatchSystemMultiImage::PatchSystemMultiImage(const std::vector<Entity> &images)
{
    for (auto img : images)
    {
        map_.emplace(img, PatchSystem{img});
    }
}

//
// std::vector<float> CudaSvmFunctionValue(const ObjectList &object_list, const SVM &svm, float gamma)
// {
//     float *c_alpha_y = nullptr;
//     float *c_x = nullptr;
//     float *c_object = nullptr;
//     float *c_f_value = nullptr;
//
//     const std::size_t n_obj = object_list.size();
//     const std::size_t n_size = svm.alpha_y_.size();
//
//     CudaAssert(cudaMalloc(&c_f_value, n_obj * n_size * sizeof(float)));
//     CudaAssert(cudaMalloc(&c_alpha_y, n_size * sizeof(float)));
//     CudaAssert(cudaMalloc(&c_x, svm.x_.size() * n_size * sizeof(float)));
//     CudaAssert(cudaMalloc(&c_object, n_obj * n_size * sizeof(float)));
//
//     CudaAssert(cudaMemcpy(c_alpha_y, svm.alpha_y_.data(), svm.alpha_y_.size() * sizeof(float), cudaMemcpyHostToDevice));
//     CudaAssert(cudaMemcpy(c_x, svm.x_.data(), svm.x_.size() * n_size * sizeof(float), cudaMemcpyHostToDevice));
//     CudaAssert(cudaMemcpy(c_object, object_list.data(), n_obj * n_size * sizeof(float), cudaMemcpyHostToDevice));
//
//     dim3 threads_division{16, 64};
//     LOG_INFO("Running Function value SVM");
//     FunctionValueSVM<<<(n_obj / 16) + 1, threads_division>>>(c_alpha_y, c_x, c_f_value, c_object, gamma, n_size, n_obj);
//     LOG_INFO("Ended Function value SVM");
//
//     cudaFree(c_alpha_y);
//     cudaFree(c_x);
//     cudaFree(c_object);
//
//     int *sum_key = nullptr;
//     CudaAssert(cudaMalloc(&sum_key, n_obj * n_size * sizeof(int)));
//     GenerateKey<<<(n_size * n_size / 1024) + 1, 1024>>>(sum_key, n_obj, n_size);
//
//     int *out_key = nullptr;
//     CudaAssert(cudaMalloc(&out_key, n_obj * n_size * sizeof(int)));
//
//     float *c_sum_f_value = nullptr;
//     CudaAssert(cudaMalloc(&c_sum_f_value, n_obj * sizeof(float)));
//     CudaAssert(cudaMemset(c_sum_f_value, 0, n_obj * sizeof(float)));
//
//     LOG_INFO("Runngin reduce");
//     thrust::reduce_by_key(thrust::device, sum_key, sum_key + n_obj * n_size, c_f_value, out_key, c_sum_f_value);
//     cudaDeviceSynchronize();
//
//     cudaFree(c_f_value);
//     cudaFree(sum_key);
//     cudaFree(out_key);
//
//     std::vector<float> result;
//     result.resize(n_obj);
//
//     CudaAssert(cudaMemcpy(result.data(), c_sum_f_value, n_obj * sizeof(float), cudaMemcpyDeviceToHost));
//
//     cudaFree(c_sum_f_value);
//
//     return result;
// }
