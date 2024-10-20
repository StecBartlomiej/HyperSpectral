#ifndef HYPERSPECTRAL_IMAGE_HPP
#define HYPERSPECTRAL_IMAGE_HPP

#include "Logger.hpp"
#include "EnviHeader.hpp"

#include <memory>
#include <filesystem>
#include <cuda_runtime.h>



inline void GpuAssert(cudaError_t code, bool abort=true)
{
    if (code != cudaSuccess)
    {
        LOG_ERROR("GPUassert: {} {} {}\n", cudaGetErrorString(code), __FILE__, __LINE__);
        if (abort) exit(code);
    }
}

[[nodiscard]] std::unique_ptr<float> LoadImage(const std::filesystem::path &path, const EnviHeader &envi);

[[nodiscard]] std::unique_ptr<float> LoadImage(std::istream &iss, const EnviHeader &envi);


[[nodiscard]] cudaPitchedPtr LoadImageCuda(const EnviHeader &envi, float* data);


#endif //HYPERSPECTRAL_IMAGE_HPP
