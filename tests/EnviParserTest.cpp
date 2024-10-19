#include <catch2/catch_test_macros.hpp>

#include "EnviParser.hpp"

#include <sstream>
#include <string_view>
#include <vector>
#include <ranges>


using namespace envi;

TEST_CASE("Parser basic usage", "[Envi]")
{
    std::vector<Token> tokens = {
            Token{TokenType::WORD, "lines"},
            Token{TokenType::EQUAL, "="},
            Token{TokenType::NUMBER, "3"},
            Token{TokenType::NEW_LINE, "\\n"},
            Token{TokenType::SEMICOLON, ";"},
            Token{TokenType::WORD, "AAA"},
            Token{TokenType::NEW_LINE, "\\n"},
            Token{TokenType::END_FILE, ""},
    };

    Parser parser{tokens};

    std::vector<Expression> expressions = Parse(parser);

    REQUIRE(expressions.size() == 1);
    REQUIRE(expressions[0].field == "lines");
    REQUIRE(std::get<int>(expressions[0].value) == 3);
}

TEST_CASE("Envi parser and lexer integration", "[Envi]")
{
    constexpr static std::string_view input = R"V0G0N( ENVI
    four words in field = 3
    ; comments should be ignored
    float = -1.5
    string = Word
    string2 = Long text in value
    string3 = Numbers in string 1 23
    list = {1, 2, 3}
    )V0G0N";

    std::stringstream stringstream{input.data()};

    EnviLexer lexer{stringstream};

    std::vector<Token> tokens;
    while (!lexer.Eof())
    {
        tokens.push_back(lexer.NextToken());
    }

    Parser parser{tokens};

    std::vector<Expression> expressions = Parse(parser);

    REQUIRE(expressions.size() == 6);

    REQUIRE(expressions[0].field == "four words in field");
    REQUIRE(std::get<int>(expressions[0].value) == 3);

    REQUIRE(expressions[1].field == "float");
    REQUIRE(std::get<float>(expressions[1].value) == -1.5f);

    REQUIRE(expressions[2].field == "string");
    REQUIRE(std::get<std::string>(expressions[2].value) == "Word");

    REQUIRE(expressions[3].field == "string2");
    REQUIRE(std::get<std::string>(expressions[3].value) == "Long text in value");

    REQUIRE(expressions[4].field == "string3");
    REQUIRE(std::get<std::string>(expressions[4].value) == "Numbers in string 1 23");

    REQUIRE(expressions[5].field == "list");
    const std::vector<std::string> result_list = {"1", "2", "3"};
    auto list = std::get<std::vector<std::string>>(expressions[5].value);

    for (std::size_t idx = 0; idx < result_list.size(); ++idx)
    {
        REQUIRE(list[idx] == result_list[idx]);
    }
}
