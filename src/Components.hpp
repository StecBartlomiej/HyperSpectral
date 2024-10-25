#ifndef COMPONENTS_HPP
#define COMPONENTS_HPP

#include <filesystem>


struct FilesystemPaths
{
    std::filesystem::path envi_header;
    std::filesystem::path img_data;
};

struct ImageSize
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

void RegisterComponents();



#endif //COMPONENTS_HPP
