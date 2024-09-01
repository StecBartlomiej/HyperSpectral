#include <catch2/catch_test_macros.hpp>

#include "EnviLexer.hpp"

#include <sstream>
#include <string_view>
#include <vector>


TEST_CASE("Basic usage", "[EnviLexer]")
{
    constexpr static std::string_view input = R"V0G0N( ENVI
    AA 1
    1.0
    -1.3
    =;,%:
    (){}
    new value = 3
    _
    )V0G0N";

    const std::vector<Token> expected_output = {
            Token{TokenType::WORD, "AA"},
            Token{TokenType::NUMBER, "1"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::NUMBER, "1.0"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::NUMBER, "-1.3"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::EQUAL, "="},
            Token{TokenType::SEMICOLON, ";"},
            Token{TokenType::COMMA, ","},
            Token{TokenType::PERCENTAGE, "%"},
            Token{TokenType::COLON, ":"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::LEFT_PARENTHESIS, "("},
            Token{TokenType::RIGHT_PARENTHESIS, ")"},
            Token{TokenType::LEFT_BRACE, "{"},
            Token{TokenType::RIGHT_BRACE, "}"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::WORD, "new"},
            Token{TokenType::WORD, "value"},
            Token{TokenType::EQUAL, "="},
            Token{TokenType::NUMBER, "3"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::UNKNOWN, "_"},
            Token{TokenType::NEW_LINE, "\\n"},

            Token{TokenType::END_FILE, ""},
    };

    std::stringstream sstream(input.data());

    EnviLexer lexer{sstream};

    Token token;
    std::size_t i = 0;
    do
    {
        token = lexer.NextToken();

        REQUIRE(i < expected_output.size());
        {
            INFO("TokenType, got: " << to_string(token.token_type) << ", expected: "\
                 << to_string(expected_output[i].token_type));
            REQUIRE(token.value == expected_output[i].value);
        }
        {
            INFO("Value, got: " << token.value << ", expected: " << expected_output[i].value);
            REQUIRE(token.token_type == expected_output[i].token_type);
        }
        ++i;
    } while (token.token_type != TokenType::END_FILE);
}

TEST_CASE("Throwing constructor", "[EnviLexer]")
{
    constexpr static std::string_view input = R"V0G0N(
    AA 1
    1.0
    -1.3
    =;,%:
    (){}
    _
    )V0G0N";

    std::stringstream sstream(input.data());

    std::unique_ptr<EnviLexer> ptr{};
    CHECK_THROWS_AS(ptr.reset(new EnviLexer{sstream}), std::runtime_error);
}
