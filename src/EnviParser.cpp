#include "EnviParser.hpp"
#include "EnviLexer.hpp"
#include "Logger.hpp"

#include <fstream>
#include <spdlog/fmt/fmt.h>


std::optional<EnviHeader> ParseEnviFile(const std::filesystem::path &path)
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


    Parser parser{tokens};
    EnviHeader envi_header{};

    while (!parser.End())
    {
        if (parser.Get().token_type == TokenType::SEMICOLON)
        {
            auto opt_new_line_idx = parser.Find(TokenType::NEW_LINE);
            if (!opt_new_line_idx.has_value())
            {
                break;
            }
            parser.JumpTo(opt_new_line_idx.value());
        }
        else if (parser.Get().token_type == TokenType::WORD)
        {
            try
            {
                ParseWord(parser, envi_header);
            }
            catch (const std::runtime_error &err)
            {
                LOG_ERROR("In parsing encountered {}. The line is ignored", err.what());
                while (parser.Get().token_type != TokenType::NEW_LINE || parser.Get().token_type != TokenType::END_FILE)
                {
                    parser.Next();
                }
            }
        }
        else if (parser.Get().token_type == TokenType::END_FILE)
        {
            break;
        }
        else
        {
            auto token = parser.Get();
            LOG_ERROR("ENVI parser has failed on token_type: {}, value: {}",
                      to_string(token.token_type), token.value);
            return std::nullopt;
        }

        parser.Next();
    }
    LOG_INFO("Parsing ended successfully");
    return std::optional<EnviHeader>{envi_header};
}

std::optional<std::size_t> Parser::Find(TokenType type) const noexcept
{
     auto iter = std::find_if(std::next(tokens_.cbegin(), static_cast<int>(pos_)),
                              tokens_.cend(),
                              [=](const Token &t){return t.token_type == type; });

     return iter != tokens_.end() ? std::optional<std::size_t>(pos_) : std::nullopt;
}

void ParseWord(Parser &parser, EnviHeader &enviHeader)
{
    auto field_value = parser.Get().value;
    parser.Next();

    if (field_value == "bands")
    {
        auto opt_value = ParseUInt(parser);
        if (!opt_value.has_value())
        {
            auto err_str = fmt::format("Expected uint value in field 'bands', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.bands_number = opt_value.value();
    }
    else if (field_value == "byte")
    {
        AssertString(parser.Get().value, "order");
        parser.Next();
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        auto opt_value = ParseUInt(parser);
        if (!opt_value.has_value() || (opt_value.value() != 0 && opt_value.value() != 1))
        {
            auto err_str = fmt::format("Expected '0' or '1' value in field 'byte order', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.byte_order = static_cast<ByteOrder>(opt_value.value());
    }
    else if (field_value == "data")
    {
        AssertString(parser.Get().value, "type");
        parser.Next();
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        auto value = ParseUInt(parser).value_or(std::numeric_limits<uint32_t>::max());
        auto opt_data_type = GetDataType(value);
        if (!opt_data_type.has_value())
        {
            auto err_str = fmt::format("Expected one of [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15] value in field"
                                       " 'data type', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.data_type =opt_data_type.value();
    }
    else if (field_value == "file")
    {
        AssertString(parser.Get().value, "type");
        parser.Next();
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        throw std::runtime_error("Field 'tile type' not implemented");
    }
    else if (field_value == "header")
    {
        AssertString(parser.Get().value, "offset");
        parser.Next();
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        auto opt_value = ParseUInt(parser);
        if (!opt_value.has_value())
        {
            auto err_str = fmt::format("Expected uint value in field 'header offset', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.header_offset = opt_value.value();
    }
    else if (field_value == "interleave")
    {
        assert(parser.Get().token_type == TokenType::WORD);
        auto opt_value = GetInterleave(parser.Get().value);
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        if (!opt_value.has_value())
        {
            auto err_str = fmt::format("Expected one of ['BIL', 'BSQ', 'BIP'] value in field "
                                       "'interleave', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.interleave = opt_value.value();
    }
    else if (field_value == "lines")
    {
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        auto opt_value = ParseUInt(parser);
        if (!opt_value.has_value())
        {
            auto err_str = fmt::format("Expected uint value in field 'lines', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.lines_per_image = opt_value.value();
    }
    else if (field_value == "samples")
    {
        assert(parser.Get().token_type == TokenType::EQUAL);
        parser.Next();

        auto opt_value = ParseUInt(parser);
        if (!opt_value.has_value())
        {
            auto err_str = fmt::format("Expected uint value in field 'samples', got {}",  parser.Get().value);
            throw std::runtime_error(err_str);
        }
        enviHeader.samples_per_image = opt_value.value();
    }
    else
    {
        auto err_str = fmt::format("Encountered unknown field {}",  parser.Get().value);
        throw std::runtime_error(err_str);
    }
    parser.Next();
}

void AssertString(std::string_view value, std::string_view expect)
{
    if (value != expect)
    {
        auto str = fmt::format("Unknown field, expected '{}' got '{}'", expect, value);
        throw std::runtime_error(str);
    }
}


std::optional<uint32_t> ParseUInt(Parser &parser) noexcept
{
    assert(parser.Get().token_type == TokenType::NUMBER);
    try
    {
        auto value = std::stol(parser.Get().value);
        return std::optional<uint32_t>{value};
    }
    catch (const std::invalid_argument &err)
    {
        return std::nullopt;
    }
}
