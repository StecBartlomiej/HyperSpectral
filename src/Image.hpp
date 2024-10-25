#ifndef HYPERSPECTRAL_IMAGE_HPP
#define HYPERSPECTRAL_IMAGE_HPP

#include "Logger.hpp"
#include "EnviHeader.hpp"
#include "Components.hpp"
#include "EntityComponentSystem.hpp"

#include <memory>
#include <filesystem>
#include <cuda_runtime.h>
#include <fstream>


[[nodiscard]] Entity CreateImage(const FilesystemPaths &paths);

[[nodiscard]] std::shared_ptr<float> LoadImage(const std::filesystem::path &path, const EnviHeader &envi);

[[nodiscard]] std::shared_ptr<float> LoadImage(std::istream &iss, const EnviHeader &envi);


class AlgorithmPCA
{
public:
    void Run();
};

[[nodiscard]] cudaPitchedPtr LoadImageCuda(const EnviHeader &envi, float* data);

template<typename T>
[[nodiscard]] std::shared_ptr<float> LoadImageType(std::istream &iss, const EnviHeader &envi)
{
    assert(envi.byte_order == ByteOrder::LITTLE_ENDIAN);

    std::shared_ptr<float> host_data{new float[envi.bands_number *
                                       envi.lines_per_image *
                                       envi.samples_per_image]};

    std::function<float*(std::size_t, std::size_t, std::size_t)> access_scheme;

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
    return std::move(host_data);
}

#endif //HYPERSPECTRAL_IMAGE_HPP
