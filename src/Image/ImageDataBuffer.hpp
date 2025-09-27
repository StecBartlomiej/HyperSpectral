#ifndef IMAGEDATABUFFER_HPP
#define IMAGEDATABUFFER_HPP


#include "Components.hpp"

#include <cuda_runtime.h>
#include <memory>


class ImageDataBuffer;

struct ImageManager
{
    std::shared_ptr<ImageDataBuffer> img;
};


enum class TileType
{
    cpu, gpu
};

template <TileType>
struct TilePtr
{
    float *ptr;

    explicit operator float*() const { return ptr; }
};

template <TileType type>
struct TileView
{
    std::size_t w, h, channels;  ///< Size of tile
    std::size_t x0, y0;          ///< Top left corner coordinates in original image
    std::size_t pitch;           ///< Allocated size of row in elements
    TilePtr<type> data;          ///< Strong typed ptr to data on host/device

    [[nodiscard]]
    __host__ __device__
    float& at(std::size_t x, std::size_t y, std::size_t ch) noexcept
    {
        assert(x < w && y < h && ch < channels);
        const auto ptr_offset = (ch + 1) * y * pitch + x;
        return data.ptr[ptr_offset];
    }

};

using GpuTile = TileView<TileType::gpu>;
using CpuTile = TileView<TileType::cpu>;


template <typename T>
concept DerivedFromInputStream = std::is_base_of_v<std::istream, std::remove_reference_t<T>>;


// TODO: use co_yield
class DataStream
{
public:
    using AccessScheme = std::function<float(std::size_t, std::size_t, std::size_t, std::istream&)>; // Should capture ImageSize

    DataStream(std::unique_ptr<std::istream>&& istream, AccessScheme access_scheme);

    [[nodiscard]]
    float GetData(std::size_t row, std::size_t col, std::size_t channel);

public:
    AccessScheme fn_access_scheme_;

private:
    std::unique_ptr<std::istream> istream_;
};

struct PagedLockBuffer
{
    explicit PagedLockBuffer(ImageSize size);
    ~PagedLockBuffer();

    // TODO: add async copy from host->device
    // TODO: add async copy from device->host

    cudaExtent extent;          ///< Holds the logical size(ImageSize) of allocated memory
    cudaPitchedPtr host_ptr;    ///< Pitched pointer to host pinned memory, has the same allocated size in bytes as \a device_ptr
    cudaPitchedPtr device_ptr;  ///< Pitched pointer to device memory allocated by \a cudaMalloc3D()
};



class ImageDataBuffer
{
public:
    [[nodiscard]]
    static ImageDataBuffer CreateFromFile(const FilesystemPaths &paths);

    ImageDataBuffer(DataStream &&data_stream, ImageSize size, cudaStream_t cuda_stream);

    // CpuTile GetCpuTile();
    //
    // GpuTile GetGpuTile();


private:
    DataStream data_stream_;
    ImageSize size_;
    cudaStream_t cuda_stream_;
    PagedLockBuffer main_buffer_;
    PagedLockBuffer tmp_buffer_;
};


#endif //IMAGEDATABUFFER_HPP
