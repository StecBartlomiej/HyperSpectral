#ifndef HYPERCPP_ENVIHEADER_HPP
#define HYPERCPP_ENVIHEADER_HPP

#include <cstdint>
#include <vector>
#include <optional>
#include <string_view>


enum class ByteOrder
{
    LITTLE_ENDIAN = 0, BIG_ENDIAN = 1
};

enum class DataType
{
    BYTE,
    INT16,
    INT32,
    FLOAT32,
    FLOAT64,
    COMPLEX32,
    COMPLEX64,
    UINT16,
    UINT32,
    INT64,
    UINT64
};

std::optional<DataType> GetDataType(uint32_t value) noexcept;

enum class Interleave
{
    BSQ, BIL, BIP
};

std::optional<Interleave> GetInterleave(std::string_view text) noexcept;

enum class LengthUnits
{
    MICRO, NANO, MILLI, CENT, METER, UNKNOWN
};

struct EnviHeader
{
    uint32_t bands_number;
    ByteOrder byte_order;
    DataType data_type;
    uint32_t header_offset;
    Interleave interleave;
    uint32_t lines_per_image;
    uint32_t samples_per_image;
    LengthUnits wavelength_unit;
    std::vector<float> wavelengths;
};


#endif //HYPERCPP_ENVIHEADER_HPP
