#include "EnviHeader.hpp"
#include "Logger.hpp"


std::optional<DataType> GetDataType(uint32_t value) noexcept
{
    switch (value)
    {
        case 1:
            return std::optional<DataType>{DataType::BYTE};
        case 2:
            return std::optional<DataType>{DataType::INT16};
        case 3:
            return std::optional<DataType>{DataType::INT32};
        case 4:
            return std::optional<DataType>{DataType::FLOAT32};
        case 5:
            return std::optional<DataType>{DataType::FLOAT64};
        case 6:
            return std::optional<DataType>{DataType::COMPLEX32};
        case 9:
            return std::optional<DataType>{DataType::COMPLEX64};
        case 12:
            return std::optional<DataType>{DataType::UINT16};
        case 13:
            return std::optional<DataType>{DataType::UINT32};
        case 14:
            return std::optional<DataType>{DataType::INT64};
        case 15:
            return std::optional<DataType>{DataType::UINT64};
        default:
            return std::optional<DataType>{std::nullopt};
    }
}

std::optional<Interleave> GetInterleave(std::string_view text) noexcept
{
    if (text == "BSQ")
        return std::optional<Interleave>{Interleave::BSQ};
    else if (text == "BIL")
        return std::optional<Interleave>{Interleave::BIL};
    else if (text == "BIP")
        return std::optional<Interleave>{Interleave::BIP};
    return std::optional<Interleave>{std::nullopt};
}
