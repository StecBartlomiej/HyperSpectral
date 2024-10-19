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

    std::vector<float> vec_result = {405.79999f, 415.39999f, 424.89999f, 434.39999f, 443.89999f,     453.5f,
                                     463.f,     472.5f,       482.f,     491.5f,       501.f, 510.60001f,
                                     520.09998f, 529.59998f, 539.09998f, 548.59998f, 558.09998f, 567.59998f,
                                     577.20001f, 586.70001f, 596.20001f, 605.70001f, 615.20001f, 624.70001f,
                                     634.20001f, 643.79999f, 653.29999f, 662.79999f, 672.29999f, 681.79999f,
                                     691.29999f, 700.79999f, 710.29999f, 719.79999f, 729.40002f, 738.90002f,
                                     748.40002f, 757.90002f, 767.40002f, 776.90002f, 786.40002f, 795.90002f,
                                     805.5f,       815.f,     824.5f,       834.f,     843.5f,       853.f,
                                     862.5f, 872.09998f, 881.59998f, 891.09998f, 900.59998f, 910.09998f,
                                     919.59998f, 929.20001f, 938.70001f, 948.20001f, 957.70001f, 967.20001f,
                                     976.79999f, 986.29999f, 995.79999f,     1005.3f};

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

    auto vec_wave = envi.wavelengths.value();
    for (std::size_t idx = 0; idx < vec_wave.size(); ++idx)
    {
        REQUIRE(vec_wave[idx] == vec_result[idx]);
    }
}

