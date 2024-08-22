#ifndef HYPERCPP_IMAGEHEADER_HPP
#define HYPERCPP_IMAGEHEADER_HPP

#include <cstdint>
#include <vector>
#include <filesystem>
#include <optional>


enum class ByteOrder
{
    BIG_ENDIAN, LITTLE_ENDIAN
};

enum class DataType
{
    FLOAT32, UNKNOWN
};

enum class Interleave
{
    BSQ, BIL, BIP
};

enum class LengthUnits
{
    MICRO, NANO, MILLI, CENT, METER, UNKNOWN
};

struct EnviHeader
{
    std::size_t bands_number;
    ByteOrder byte_order;
    DataType data_type;
    std::size_t header_offset;
    Interleave interleave;
    std::size_t lines_per_image;
    std::size_t samples_per_image;
    LengthUnits wavelength_unit;
    std::vector<float> wavelengths;
};

[[nodiscard]] std::optional<EnviHeader> ParseEnvi(const std::filesystem::path &path);





#endif //HYPERCPP_IMAGEHEADER_HPP
