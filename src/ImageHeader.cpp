#include "ImageHeader.hpp"
#include "Logger.hpp"

#include <fstream>
#include <string>
#include <algorithm>




std::optional<EnviHeader> ParseEnvi(const std::filesystem::path &path)
{
    std::ifstream file{path};

    std::string line;
    std::getline(file, line);

    if (line == "ENVI")
    {
        LOG_CRITICAL("File {} does not contain 'ENVI' header in first line, content is ignored.", path.string());
        return std::nullopt;
    }

    std::size_t line_number = 1;
    EnviHeader envi_header{}; // ?

    while (std::getline(file, line))
    {
        // ignore comments
        if (line.starts_with(';'))
            continue;

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos)
        {
            LOG_WARN("In file {}, on line {}, did not found '=', line is ignored", path.string(), line_number);
            continue;
        }
        else if (line[eq_pos - 1] != ' ')
        {
            LOG_ERROR("In file {}, on line {}, did not found space before '=', line is ignored", path.string(), line_number);
            continue;
        }

        std::string_view keyword{line.substr(0, eq_pos - 1)};

        // Parse keyword
        if (keyword == "bands")
        {

        }
        else if (keyword == "byte order")
        {

        }


        ++line_number;
    }
    return std::optional<EnviHeader>{envi_header};
}
