#include <catch2/catch_test_macros.hpp>

#include "EnviHeader.hpp"

#include <sstream>
#include <string_view>
#include <vector>


TEST_CASE("Parse envi header", "[Envi]")
{
    constexpr static std::string_view input = R"V0G0N( ENVI
    bands = 64
    data type = 4
    interleave = bsq
    header offset = 0
    wavelength units = Nanometers
    byte order = 0
    lines = 325
    samples = 220
    )V0G0N";


    std::stringstream stream{input.data()};
    auto opt_envi = ParseEnviText(stream);

    REQUIRE(opt_envi.has_value());

    auto envi = opt_envi.value();

    REQUIRE(envi.bands_number == 64);
    REQUIRE(envi.data_type == DataType::FLOAT32);
    REQUIRE(envi.interleave == Interleave::BSQ);
    REQUIRE(envi.header_offset == 0);
    REQUIRE(envi.wavelength_unit == LengthUnits::NANO);
    REQUIRE(envi.byte_order == ByteOrder::LITTLE_ENDIAN);
    REQUIRE(envi.lines_per_image == 325);
    REQUIRE(envi.samples_per_image == 220);
}

