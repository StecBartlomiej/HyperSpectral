#include "ImageDataBuffer.hpp"


#include <source_location>


DataStream::DataStream(std::unique_ptr<std::istream>&& istream, AccessScheme access_scheme): istream_(std::move(istream)), fn_access_scheme_(access_scheme)
{
}

float DataStream::GetData(std::size_t row, std::size_t col, std::size_t channel)
{
    return fn_access_scheme_(row, col, channel, *istream_);
}


PagedLockBuffer::PagedLockBuffer(ImageSize size): extent(make_cudaExtent(size.width * sizeof(float), size.height, size.channel))
{
    // Device
    cudaError_t error = cudaMalloc3D(&device_ptr, extent);

    if (error != cudaSuccess)
    {
        const auto loc = std::source_location::current();
        LOG_ERROR("File: {}: '{}', cudaMalloc3D failed with error code: {}\n",
            loc.file_name(), loc.function_name(), cudaGetErrorString(error));

        throw std::runtime_error("PagedLockBuffer: cudaMalloc3D failed!");
    }

    // Host
    float* pinned_ptr = nullptr;
    const auto memory_byte_size = device_ptr.pitch * extent.height * extent.depth * sizeof(float);

    error = cudaMallocHost(&pinned_ptr, memory_byte_size);

    if (error != cudaSuccess)
    {
        cudaFree(device_ptr.ptr);
        const auto loc = std::source_location::current();
        LOG_ERROR("File: {}: '{}', cudaMallocHost failed with error code: {}\n",
            loc.file_name(), loc.function_name(), cudaGetErrorString(error));

        throw std::runtime_error("PagedLockBuffer: cudaMallocHost failed!");
    }
    host_ptr = make_cudaPitchedPtr(pinned_ptr, device_ptr.pitch, extent.width, extent.height);
}

PagedLockBuffer::~PagedLockBuffer()
{
    cudaFreeHost(host_ptr.ptr);
    cudaFree(device_ptr.ptr);
}



ImageDataBuffer::ImageDataBuffer(DataStream&& data_stream, ImageSize size, cudaStream_t cuda_stream):
    data_stream_{std::move(data_stream)}, size_{size}, cuda_stream_{cuda_stream}, main_buffer_(size), tmp_buffer_(size)
{

}

