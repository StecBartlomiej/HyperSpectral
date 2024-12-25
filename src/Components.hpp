#ifndef COMPONENTS_HPP
#define COMPONENTS_HPP

#include <filesystem>
#include <cereal/cereal.hpp>


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

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            CEREAL_NVP(width),
            CEREAL_NVP(height),
            CEREAL_NVP(depth));
    }
};


[[nodiscard]] bool operator==(const ImageSize &lhs, const ImageSize &rhs) noexcept;

void RegisterComponents();



#endif //COMPONENTS_HPP
