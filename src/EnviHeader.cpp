#include "EnviHeader.hpp"
#include "Logger.hpp"
#include "EnviParser.hpp"

#include <fstream>


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

std::optional<LengthUnits> GetLengthUnits(std::string_view text) noexcept
{
    if (text == "Micrometers" || text == "um")
        return std::optional<LengthUnits>{LengthUnits::MICRO};
    else if (text == "Nanometers" || text == "nm")
        return std::optional<LengthUnits>{LengthUnits::NANO};
    else if (text == "Millimeters" || text == "mm")
        return std::optional<LengthUnits>{LengthUnits::MILLI};
    else if (text == "Centimeters" || text == "cm")
        return std::optional<LengthUnits>{LengthUnits::CENT};
    return std::optional<LengthUnits>{std::nullopt};
}


std::optional<EnviHeader> ParseEnvi(const std::filesystem::path &path)
{
    LOG_INFO("Parsing file {}", path.string());

    std::ifstream file{path};

    if (!file.is_open())
    {
        LOG_ERROR("Cant open open file {} to parse ENVI HEADER", path.string());
        return std::nullopt;
    }

    return ParseEnviText(file);
}


std::optional<EnviHeader> ParseEnviText(std::istream &iss)
{
    LOG_INFO("Envi parsing starts");
    std::vector<Token> tokens;
    try
    {
        EnviLexer lexer{iss};
        while (!lexer.Eof())
        {
            tokens.push_back(lexer.NextToken());
        }
    }
    catch (const std::runtime_error &err)
    {
        LOG_ERROR("While tokenizing EnviHeader, {}", err.what());
    }
    LOG_INFO("Envi Lexer exited successfully");


//    Parser parser{tokens};

    LOG_INFO("Parsing ended successfully");
//    return std::optional<EnviHeader>{envi_header};
    return std::nullopt;
}

