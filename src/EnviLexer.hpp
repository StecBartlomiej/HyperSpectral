#ifndef HYPERCPP_ENVILEXER_HPP
#define HYPERCPP_ENVILEXER_HPP

#include <string>
#include <fstream>
#include <filesystem>
#include <optional>
#include <functional>
#include <string_view>


enum class TokenType
{
    WORD,
    NUMBER,

    EQUAL,
    SEMICOLON,
    COMMA,
    PERCENTAGE,
    COLON,

    LEFT_BRACE,
    RIGHT_BRACE,
    LEFT_PARENTHESIS,
    RIGHT_PARENTHESIS,

    NEW_LINE,
    END_FILE,
    UNKNOWN
};


struct Token
{
    TokenType token_type;
    std::string value;
};

class EnviLexer
{
public:
    explicit EnviLexer(std::istream &iss);

    [[nodiscard]] Token NextToken();

    [[nodiscard]] char GetChar() const { return ch_; }
    [[nodiscard]] std::optional<char> PeekChar();
    [[nodiscard]] bool eof() const { return iss_.eof(); }

    void ReadChar();

private:
    std::istream &iss_;
    char ch_;
};

[[nodiscard]] std::string ReadNumber(EnviLexer &lexer);

void SkipSpaces(EnviLexer &lexer);

[[nodiscard]] bool IsLetter(char c);

[[nodiscard]] std::string ParseDigit(EnviLexer &lexer);

template <bool (*Function)(char)>
[[nodiscard]] std::string ReadWhileTrue(EnviLexer &lexer)
{
    std::ostringstream oss;
    while (!lexer.eof())
    {
        oss << lexer.GetChar();

        auto opt_value = lexer.PeekChar();
        if (!Function(opt_value.value_or('\0')))
            break;

        lexer.ReadChar();
    }
    return oss.str();
}

[[nodiscard]] constexpr std::string_view to_string(TokenType token_type) noexcept
{
    switch (token_type)
    {
        case TokenType::WORD:
            return "WORD";
        case TokenType::NUMBER:
            return "NUMBER";
        case TokenType::EQUAL:
            return "EQUAL";
        case TokenType::SEMICOLON:
            return "SEMICOLON";
        case TokenType::COMMA:
            return "COMMA";
        case TokenType::PERCENTAGE:
            return "PERCENTAGE";
        case TokenType::COLON:
            return "COLON";
        case TokenType::LEFT_BRACE:
            return "LEFT_BRACE";
        case TokenType::RIGHT_BRACE:
            return "RIGHT_BRACE";
        case TokenType::LEFT_PARENTHESIS:
            return "LEFT_PARENTHESIS";
        case TokenType::RIGHT_PARENTHESIS:
            return "RIGHT_PARENTHESIS";
        case TokenType::NEW_LINE:
            return "NEW_LINE";
        case TokenType::END_FILE:
            return "END_FILE";
        case TokenType::UNKNOWN:
            return "UNKNOWN";
    }
    return "";
}

#endif //HYPERCPP_ENVILEXER_HPP