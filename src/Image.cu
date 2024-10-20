#include "Image.hpp"

#include <cassert>
#include <functional>
#include <fstream>


std::unique_ptr<float> LoadImage(const std::filesystem::path &path, const EnviHeader &envi)
{
    std::ifstream file{path, std::ios_base::binary | std::ios::in};
    assert(file.is_open());
    return LoadImage(file, envi) ;
}

// TODO: add different types
std::unique_ptr<float> LoadImage(std::istream &iss, const EnviHeader &envi)
{
    assert(envi.data_type == DataType::FLOAT32);
    assert(envi.byte_order == ByteOrder::LITTLE_ENDIAN);

    std::unique_ptr<float> host_data{new float[envi.bands_number *
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
            access_scheme = [&, bands=dim1, lines=dim2, samples=dim3](std::size_t i, std::size_t j, std::size_t k) -> float* {
                return host_data.get() + i * bands + j * lines + k * samples;
            };
            break;
        case Interleave::BIP:
            dim1 = envi.lines_per_image;
            dim2 = envi.samples_per_image;
            dim3 = envi.bands_number;
            access_scheme = [&, bands=dim1, lines=dim2, samples=dim3](std::size_t i, std::size_t j, std::size_t k) -> float* {
                return host_data.get() + k * bands + i * lines + j * samples;
            };
            break;
        case Interleave::BIL:
            dim1 = envi.lines_per_image;
            dim2 = envi.bands_number;
            dim3 = envi.samples_per_image;
            access_scheme = [&, bands=dim1, lines=dim2, samples=dim3](std::size_t i, std::size_t j, std::size_t k) -> float* {
                return host_data.get() + j * bands + i * lines + k * samples;
            };
            break;
    }

    // TODO: add bit order
    for (std::size_t i = 0; i < dim1; ++i)
    {
        for (std::size_t j = 0; j < dim2; ++j)
        {
            for (std::size_t k = 0; k < dim3; ++k)
            {
                iss.read(reinterpret_cast<char*>(access_scheme(i, j, k)), sizeof(float));
            }
        }
    }



    return std::move(host_data);
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

