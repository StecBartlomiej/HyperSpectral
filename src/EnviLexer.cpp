#include "EnviLexer.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <fstream>


EnviLexer::EnviLexer(std::istream &iss): iss_{iss}, ch_{}
{
    std::string line{};
    std::getline(iss_, line);

    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

    if (line != "ENVI")
    {
        throw std::runtime_error("Missing 'ENVI' string at the beginning of the file");
    }

    ReadChar();
}

void EnviLexer::ReadChar()
{
    if (Eof())
    {
        ch_ = 0;
    }
    else
    {
        iss_.get(ch_);
    }
}

std::optional<char> EnviLexer::PeekChar()
{
    auto next_ch = iss_.peek();
    return next_ch == std::char_traits<char>::eof() ? std::nullopt : std::optional<char>{next_ch};
}

Token EnviLexer::NextToken()
{
    TokenType token_type = TokenType::UNKNOWN;
    std::string value;

    SkipSpaces(*this);

    switch (ch_)
    {
        case '\0':
            token_type = TokenType::END_FILE;
            value = "";
            break;
        case '\n':
            token_type = TokenType::NEW_LINE;
            value = "\\n";
            break;
        case ':':
            token_type = TokenType::COLON;
            value = ":";
            break;
        case '=':
            token_type = TokenType::EQUAL;
            value = "=";
            break;
        case ';':
            token_type = TokenType::SEMICOLON;
            value = ";";
            break;
        case '%':
            token_type = TokenType::PERCENTAGE;
            value = "%";
            break;
        case ',':
            token_type = TokenType::COMMA;
            value = ",";
            break;
        case '(':
            token_type = TokenType::LEFT_PARENTHESIS;
            value = "(";
            break;
        case ')':
            token_type = TokenType::RIGHT_PARENTHESIS;
            value = ")";
            break;
        case '{':
            token_type = TokenType::LEFT_BRACE;
            value = "{";
            break;
        case '}':
            token_type = TokenType::RIGHT_BRACE;
            value = "}";
            break;
        default:
            if (std::isdigit(ch_) || (ch_ == '-' && std::isdigit(PeekChar().value_or('\0'))) )
            {
                token_type = TokenType::NUMBER;
                value = ParseDigit(*this);
            }
            else if (IsLetter(ch_))
            {
                token_type = TokenType::WORD;
                value = ReadWhileTrue<IsLetter>(*this);
            }
            else
            {
                token_type = TokenType::UNKNOWN;
                value = std::string(1, ch_);
                LOG_ERROR("Unknown token {}", ch_);
            }
            break;
    }
    ReadChar();
    return Token{token_type, value};
}

void SkipSpaces(EnviLexer &lexer)
{
    while(lexer.GetChar() == ' ')
    {
        lexer.ReadChar();
    }
}

bool IsLetter(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

std::string ParseDigit(EnviLexer &lexer)
{
    std::ostringstream oss{};
    bool had_comma = false;

    while (!lexer.Eof())
    {
        oss << lexer.GetChar();

        char next_ch = lexer.PeekChar().value_or('\0');

        if (next_ch == '.')
        {
            if (had_comma)
                break;
            had_comma = true;
        }
        else if (!std::isdigit(next_ch))
        {
            break;
        }

        lexer.ReadChar();
    }
    return oss.str();
}
