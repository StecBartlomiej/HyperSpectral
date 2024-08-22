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
    if (eof())
        ch_ = 0;
    else
        iss_.get(ch_);
}

std::optional<char> EnviLexer::PeekChar()
{
    auto next_ch = iss_.peek();
    return next_ch == std::char_traits<char>::eof() ? std::nullopt : std::optional<char>{next_ch};
}

Token EnviLexer::NextToken()
{
    Token token{};

    SkipSpaces(*this);

    switch (ch_)
    {
        case '\0':
            token = Token{TokenType::END_FILE, ""};
            break;
        case '\n':
            token = Token{TokenType::NEW_LINE, "\\n"};
            break;
        case ':':
            token = Token{TokenType::COLON, ":"};
            break;
        case '=':
            token = Token{TokenType::EQUAL, "="};
            break;
        case ';':
            token = Token{TokenType::SEMICOLON, ";"};
            break;
        case '%':
            token = Token{TokenType::PERCENTAGE, "%"};
            break;
        case ',':
            token = Token{TokenType::COMMA, ","};
            break;
        case '(':
            token = Token{TokenType::LEFT_PARENTHESIS, "("};
            break;
        case ')':
            token = Token{TokenType::RIGHT_PARENTHESIS, ")"};
            break;
        case '{':
            token = Token{TokenType::LEFT_BRACE, "{"};
            break;
        case '}':
            token = Token{TokenType::RIGHT_BRACE, "}"};
            break;
        default:
            if (std::isdigit(ch_) || (ch_ == '-' && std::isdigit(PeekChar().value_or('\0'))) )
            {
                token = Token{TokenType::NUMBER, ParseDigit(*this)};
            }
            else if (IsLetter(ch_))
            {
                token = Token{TokenType::WORD, ReadWhileTrue<IsLetter>(*this)};
            }
            else
            {
                token = Token{TokenType::UNKNOWN, std::string(1, ch_)};
                LOG_ERROR("Unknown token {}", ch_);
            }
            break;
    }
    ReadChar();
    return token;
}

void SkipSpaces(EnviLexer &lexer)
{
    while(lexer.GetChar() == ' ')
        lexer.ReadChar();
}

bool IsLetter(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

std::string ParseDigit(EnviLexer &lexer)
{
    std::ostringstream oss{};
    bool had_comma = false;

    while (!lexer.eof())
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
