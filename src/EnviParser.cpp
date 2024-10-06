#include "EnviParser.hpp"
#include "EnviLexer.hpp"
#include "Logger.hpp"

#include <fstream>
#include <charconv>


namespace envi
{

bool Accept(Parser &parser, TokenType type)
{
    if (parser.Get().token_type == type)
    {
        parser.Next();
        return true;
    }
    return false;
}

bool Expect(Parser &parser, TokenType type)
{
    return parser.Get().token_type == type;
}

void SkipToNewLine(Parser &parser)
{
    while (!parser.End())
    {
        if (parser.Get().token_type == TokenType::NEW_LINE)
            return;
        parser.Next();
    }
}

auto ParseField(Parser &parser) -> std::optional<std::string>
{
    std::ostringstream oss{};
    bool insert_space = false;

    while (!Accept(parser, TokenType::EQUAL))
    {
        if (insert_space)
            oss << " ";

        if (!Expect(parser, TokenType::WORD))
        {
            LOG_ERROR("While parsing filed name, expected word, got type: {}, with value: {}",
                      to_string(parser.Get().token_type),
                      parser.Get().value);
            return std::nullopt;
        }

        oss << parser.Get().value;
        parser.Next();

        insert_space = true;
    }
    return std::optional<std::string>{oss.str()};
}

Value ParseWord(Parser &parser)
{
    std::ostringstream oss{};
    bool insert_space = false;

    while (Expect(parser, TokenType::WORD) || Expect(parser, TokenType::NUMBER))
    {
        if (insert_space)
            oss << " ";

        oss << parser.Get().value;
        parser.Next();

        insert_space = true;
    }
    return oss.str();
}

auto ParseNumber(Parser &parser) -> std::optional<Value>
{
    std::string_view str = parser.Get().value;

    auto iter = std::find(str.cbegin(), str.cend(), '.');

    if (iter == str.end())
    {
        int number;
        auto [_, ec] = std::from_chars(str.begin(), str.end(), number);

        if (ec == std::errc())
        {
            parser.Next();
            return std::optional<Value>{number};
        }
        else if (ec == std::errc::invalid_argument)
        {
            LOG_ERROR("While parsing int, expected number value got {}, with type {}",
                      parser.Get().value,
                      to_string(parser.Get().token_type));
        }
        else if (ec == std::errc::result_out_of_range)
        {
            LOG_ERROR("While parsing int, numeric value {} out of range",
                      parser.Get().value);
        }
        return std::nullopt;
    }
    else
    {
        float number;
        auto [_, ec] = std::from_chars(str.begin(), str.end(), number);

        if (ec == std::errc())
        {
            parser.Next();
            return std::optional<Value>{number};
        }
        else if (ec == std::errc::invalid_argument)
        {
            LOG_ERROR("While parsing float, expected number value got {}, with type {}",
                      parser.Get().value,
                      to_string(parser.Get().token_type));
        }
        else if (ec == std::errc::result_out_of_range)
        {
            LOG_ERROR("While parsing float, numeric value {} out of range",
                      parser.Get().value);
        }
        return std::nullopt;
    }
}

auto ParseVector(Parser &parser) -> std::vector<std::string>
{
    std::vector<std::string> values;
    parser.Next();

    while (!parser.End())
    {
        if (Accept(parser, TokenType::NEW_LINE))
            continue;
        else if (!Expect(parser, TokenType::WORD) && !Expect(parser, TokenType::NUMBER))
        {
            LOG_ERROR("While parsing list of values, expected word or number, got {} with type {}",
                      parser.Get().value,
                      to_string(parser.Get().token_type));
        }
        values.push_back(parser.Get().value);
        parser.Next();

        if (Accept(parser, TokenType::RIGHT_BRACE))
            break;
        else if (!Accept(parser, TokenType::COMMA))
        {
            LOG_ERROR("While parsing list of values, expected comma between values, got {} with type {}",
                      parser.Get().value,
                      to_string(parser.Get().token_type));
        }
    }
    return values;
}

auto ParseValue(Parser &parser) -> std::optional<Value>
{
    if (Expect(parser, TokenType::LEFT_BRACE))
    {
        return std::optional<Value>{ParseVector(parser)};
    }
    else if (Expect(parser, TokenType::NUMBER))
    {
        return ParseNumber(parser);
    }
    else if (Expect(parser, TokenType::WORD))
    {
        return std::optional<Value>{ParseWord(parser)};
    }
    LOG_ERROR("While parsing value, encountered unexpected token type: {}, with value: {}",
              to_string(parser.Get().token_type),
              parser.Get().value);
    return std::nullopt;
}

auto ParseExpression(Parser &parser) -> std::optional<Expression>
{
    auto opt_field = ParseField(parser);
    if (!opt_field.has_value())
    {
        LOG_ERROR("While parsing expressions missing field name");
        return std::nullopt;
    }

    auto opt_value = ParseValue(parser);
    if (!opt_value.has_value())
    {
        LOG_ERROR("While parsing expressions missing field value");
        return std::nullopt;
    }

    return std::optional<Expression>{Expression{opt_field.value(), opt_value.value()}};
}

auto Parse(Parser &parser) -> std::vector<Expression>
{
    std::vector<Expression> expressions;

    while (!parser.End())
    {
        if (Accept(parser, TokenType::END_FILE))
        {
            break;
        }
        else if (Accept(parser, TokenType::SEMICOLON))
        {
            SkipToNewLine(parser);
        }
        else if (Expect(parser, TokenType::WORD))
        {
            auto opt_expr = ParseExpression(parser);
            if (!opt_expr.has_value())
            {
                LOG_ERROR("Parsing expression was unsuccessful, skipping to new line");
                SkipToNewLine(parser);
                continue;
            }
            expressions.push_back(opt_expr.value());
        }
        else
        {
            LOG_ERROR("Expected expression, comment or end file got {} with value {}. Skipping to new line",
                      to_string(parser.Get().token_type),
                      parser.Get().value);
            SkipToNewLine(parser);
        }
        parser.Next();
    }
    LOG_INFO("Envi parsing exited successfully");
    return expressions;
}
}