#ifndef HYPERCPP_ENVIPARSER_HPP
#define HYPERCPP_ENVIPARSER_HPP

#include "EnviLexer.hpp"

#include <optional>
#include <variant>
#include <vector>
#include <string>


namespace envi
{

using Value = std::variant<int, float, std::string, std::vector<std::string>>;

struct Expression
{
    std::string field;
    Value value;
};

class Parser;

[[nodiscard]] bool Accept(Parser &parser, TokenType type);
[[nodiscard]] bool Expect(Parser &parser, TokenType type);

void SkipToNewLine(Parser &parser);

[[nodiscard]] auto ParseField(Parser &parser) -> std::optional<std::string>;

[[nodiscard]] Value ParseWord(Parser &parser);

[[nodiscard]] auto ParseNumber(Parser &parser) -> std::optional<Value>;

[[nodiscard]] auto ParseVector(Parser &parser) -> std::vector<std::string>;

[[nodiscard]] auto ParseValue(Parser &parser) -> std::optional<Value>;

[[nodiscard]] auto ParseExpression(Parser &parser) -> std::optional<Expression>;

[[nodiscard]] auto Parse(Parser &parser) -> std::vector<Expression>;


class Parser
{
public:
    explicit Parser(std::vector<Token> tokens): tokens_{std::move(tokens)}, pos_{0} {}

    [[nodiscard]] const Token& Get() const noexcept { return tokens_[pos_]; }

    [[nodiscard]] bool End() const { return pos_ >= tokens_.size(); }

    void Next() { ++pos_; }

private:
    std::vector<Token> tokens_;
    std::size_t pos_;
};

}

#endif //HYPERCPP_ENVIPARSER_HPP
