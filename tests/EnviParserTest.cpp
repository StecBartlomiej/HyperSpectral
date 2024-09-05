#include <catch2/catch_test_macros.hpp>

#include "EnviParser.hpp"

#include <sstream>
#include <string_view>
#include <vector>


TEST_CASE("Parser basic usage", "[ParseEnviText]")
{
    constexpr static std::string_view input = R"V0G0N( ENVI
    lines = 3
    )V0G0N";

    std::stringstream stringstream{input.data()};
    auto opt_envi_header = ParseEnviText(stringstream);

    REQUIRE(opt_envi_header.has_value());

    auto envi_header = opt_envi_header.value();
    REQUIRE(envi_header.lines_per_image == 3u);
}