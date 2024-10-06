#include "EnviHeader.hpp"
#include "Logger.hpp"
#include "EnviParser.hpp"

#include <fstream>
#include <functional>

using namespace envi;

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
    if (text == "BSQ" || text == "bsq")
        return std::optional<Interleave>{Interleave::BSQ};
    else if (text == "BIL" || text == "bil")
        return std::optional<Interleave>{Interleave::BIL};
    else if (text == "BIP" || text == "bip")
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


std::optional<EnviHeader> LoadEnvi(const std::filesystem::path &path)
{
    LOG_INFO("Loading file {}", path.string());

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


    Parser parser{tokens};
    std::vector<Expression> expressions = Parse(parser);

    EnviHeader envi_header{};

    for (const auto &expr : expressions)
    {
        try
        {
            MatchExpression(expr, envi_header);
        }
        catch (const std::runtime_error &err)
        {
            LOG_ERROR("While matching expression after parsing, {}", err.what());
            continue; // TODO: Throw/std::nullopt ?
        }
    }
    LOG_INFO("Parsing ended successfully");
    return std::optional<EnviHeader>{envi_header};
}

void MatchExpression(const Expression &expression, EnviHeader &envi_header)
{
    auto VariantToValue = [&]<typename T>(std::string_view err_msg)
    {
        auto opt_val = GetType<T>(expression.value);
        if (!opt_val.has_value())
        {
            throw std::runtime_error{err_msg.data()};
        }
        return opt_val.value();
    };

    if (expression.field == "bands")
    {
        envi_header.bands_number = VariantToValue.operator()<int>("Wrong type, expected type 'int' in field 'bands'");
    }
    else if (expression.field == "byte order")
    {
        int value = VariantToValue.operator()<int>("Wrong type, expected type 'int' in field 'byte order'");
        if (value != 0 && value != 1)
        {
            throw std::runtime_error("Wrong value in field 'byte order', expected '0' or '1'");
        }
        envi_header.byte_order = static_cast<ByteOrder>(value);
    }
    else if (expression.field == "data type")
    {
        int value = VariantToValue.operator()<int>("Wrong type, expected type 'int' in field 'data type'");
        auto opt_data_type = GetDataType(value);
        if (!opt_data_type.has_value())
        {
            throw std::runtime_error{"Wrong value in field 'data type', value does not match any data type value"};
        }
        envi_header.data_type = opt_data_type.value();
    }
    else if (expression.field == "file type")
    {
        envi_header.file_type = VariantToValue.operator()<std::string>("Wrong type, expected type 'string' in field 'file type'");
    }
    else if (expression.field == "header offset")
    {
        int value = VariantToValue.operator()<int>("Wrong type, expected type 'int' in field 'header offset'");
        if (value < 0)
        {
            throw std::runtime_error{"Value in filed 'header offset' must be a non-negative number"};
        }
        envi_header.header_offset = value;
    }
    else if (expression.field == "interleave")
    {
        auto value = VariantToValue.operator()<std::string>("Wrong type, expected type 'string' in field 'interleave'");

        auto opt_interleave = GetInterleave(value);
        if (!opt_interleave.has_value())
        {
            throw std::runtime_error{"Wrong string value in field 'interleave'"};
        }
        envi_header.interleave = opt_interleave.value();
    }
    else if (expression.field == "lines")
    {
        auto value = VariantToValue.operator()<int>("Wrong type, expected type 'int' in field 'lines'");
        if (value < 0)
        {
            throw std::runtime_error{"Value in field 'lines' must be non-negative"};
        }
        envi_header.lines_per_image = value;
    }
    else if (expression.field == "samples")
    {
        auto value = VariantToValue.operator()<int>("Wrong type, expected type 'int' in field 'samples'");
        if (value < 0)
        {
            throw std::runtime_error{"Value in field 'samples' must be non-negative"};
        }
        envi_header.samples_per_image = value;
    }
    else if (expression.field == "wavelength units")
    {
        auto value = VariantToValue.operator()<std::string>("Wrong type, expected type 'string' in field 'wavelength units'");
        auto opt_unit = GetLengthUnits(value);
        if (!opt_unit.has_value())
        {
            throw std::runtime_error{"Value in field 'wavelength units' does not match known measure unit"};
        }
        envi_header.wavelength_unit = opt_unit.value();
    }
    else
    {
        throw std::runtime_error{"Encountered unsupported field \'" + expression.field + "\'"};
    }
}
