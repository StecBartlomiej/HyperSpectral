#include <catch2/catch_test_macros.hpp>

#include "EnviHeader.hpp"

#include <sstream>
#include <string_view>
#include <vector>
#include <ranges>


TEST_CASE("Parse envi header", "[Envi]")
{
    constexpr static std::string_view input = R"V0G0N( ENVI
    bands = 64
    data type = 4
    interleave = bsq
    header offset = 0
    wavelength units = Nanometers
    byte order = 0
    wavelength = {405.79999, 415.39999, 424.89999, 434.39999, 443.89999,     453.5,
          463,     472.5,       482,     491.5,       501, 510.60001,
    520.09998, 529.59998, 539.09998, 548.59998, 558.09998, 567.59998,
    577.20001, 586.70001, 596.20001, 605.70001, 615.20001, 624.70001,
    634.20001, 643.79999, 653.29999, 662.79999, 672.29999, 681.79999,
    691.29999, 700.79999, 710.29999, 719.79999, 729.40002, 738.90002,
    748.40002, 757.90002, 767.40002, 776.90002, 786.40002, 795.90002,
        805.5,       815,     824.5,       834,     843.5,       853,
        862.5, 872.09998, 881.59998, 891.09998, 900.59998, 910.09998,
    919.59998, 929.20001, 938.70001, 948.20001, 957.70001, 967.20001,
    976.79999, 986.29999, 995.79999,     1005.3}
    lines = 325
    samples = 220
    )V0G0N";

    std::vector<float> vec_result = {405.79999, 415.39999, 424.89999, 434.39999, 443.89999,     453.5,
                                     463,     472.5,       482,     491.5,       501, 510.60001,
                                     520.09998, 529.59998, 539.09998, 548.59998, 558.09998, 567.59998,
                                     577.20001, 586.70001, 596.20001, 605.70001, 615.20001, 624.70001,
                                     634.20001, 643.79999, 653.29999, 662.79999, 672.29999, 681.79999,
                                     691.29999, 700.79999, 710.29999, 719.79999, 729.40002, 738.90002,
                                     748.40002, 757.90002, 767.40002, 776.90002, 786.40002, 795.90002,
                                     805.5,       815,     824.5,       834,     843.5,       853,
                                     862.5, 872.09998, 881.59998, 891.09998, 900.59998, 910.09998,
                                     919.59998, 929.20001, 938.70001, 948.20001, 957.70001, 967.20001,
                                     976.79999, 986.29999, 995.79999,     1005.3};

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


    REQUIRE(envi.wavelengths.has_value());
    for (auto const [idx, number] : std::views::enumerate(envi.wavelengths.value()))
    {
        REQUIRE(number == vec_result[idx]);
    }
}

