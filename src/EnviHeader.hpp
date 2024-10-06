#ifndef HYPERCPP_ENVIHEADER_HPP
#define HYPERCPP_ENVIHEADER_HPP

#include "EnviParser.hpp"

#include <cstdint>
#include <vector>
#include <optional>
#include <string_view>
#include <filesystem>


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
    MICRO, NANO, MILLI, CENT, METER
};

std::optional<LengthUnits> GetLengthUnits(std::string_view text) noexcept;

struct EnviHeader
{
    uint32_t bands_number;
    ByteOrder byte_order;
    DataType data_type;
    uint32_t header_offset;
    Interleave interleave;
    uint32_t lines_per_image;
    uint32_t samples_per_image;
    std::string file_type;

    std::optional<LengthUnits> wavelength_unit;
    std::optional<std::vector<float>> wavelengths;
};

[[nodiscard]] std::optional<EnviHeader> LoadEnvi(const std::filesystem::path &path);

[[nodiscard]] std::optional<EnviHeader> ParseEnviText(std::istream &iss);

void MatchExpression(const envi::Expression &expression, EnviHeader &envi_header);

template <typename T>
std::optional<T> GetType(envi::Value value) noexcept
{
    if (!std::holds_alternative<T>(value))
    {
        return std::nullopt;
    }
    return std::optional<T>{std::get<T>(value)};
}

#endif //HYPERCPP_ENVIHEADER_HPP
