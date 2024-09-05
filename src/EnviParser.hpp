#ifndef HYPERCPP_ENVIPARSER_HPP
#define HYPERCPP_ENVIPARSER_HPP

#include "EnviHeader.hpp"
#include "EnviLexer.hpp"

#include <optional>
#include <filesystem>
#include <cassert>
#include <expected>
#include <exception>


[[nodiscard]] std::optional<EnviHeader> ParseEnviFile(const std::filesystem::path &path);

[[nodiscard]] std::optional<EnviHeader> ParseEnviText(std::istream &iss);

class Parser
{
public:
    explicit Parser(std::vector<Token> tokens): tokens_{std::move(tokens)}, pos_{0} {}

    [[nodiscard]] const Token& Get() const noexcept { return tokens_[pos_]; }

    [[nodiscard]] bool End() const { return pos_ >= tokens_.size(); }

    void Next() { ++pos_; }

    [[nodiscard]] std::optional<std::size_t> Find(TokenType type) const noexcept;

    void JumpTo(std::size_t idx) { assert(idx < tokens_.size()); pos_ = idx; }

private:
    std::vector<Token> tokens_;
    std::size_t pos_;
};

void AssertString(std::string_view value, std::string_view expect);

void ParseWord(Parser &parser, EnviHeader &enviHeader);

[[nodiscard]] std::optional<uint32_t> ParseUInt(Parser &parser) noexcept;

#endif //HYPERCPP_ENVIPARSER_HPP
