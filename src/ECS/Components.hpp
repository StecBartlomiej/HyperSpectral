#ifndef COMPONENTS_HPP
#define COMPONENTS_HPP

#include "EntityComponentSystem.hpp"

#include <memory>
#include <filesystem>
#include <cereal/cereal.hpp>


struct FilesystemPaths
{
    std::filesystem::path envi_path;
    std::filesystem::path img_path;
};

struct ImageSize
{
    std::size_t width;
    std::size_t height;
    std::size_t channel;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
            CEREAL_NVP(width),
            CEREAL_NVP(height),
            CEREAL_NVP(channel));
    }
};


[[nodiscard]]
std::size_t FlattenIdx(ImageSize img, std::size_t channel, std::size_t height, std::size_t width);

struct PatchData
{
    std::size_t center_x;
    std::size_t center_y;
    constexpr static std::size_t S = 9;
};

struct PatchLabel
{
    PatchData patch;
    Entity img;
};

[[nodiscard]]
bool operator==(const ImageSize &lhs, const ImageSize &rhs) noexcept;

void RegisterComponents();



#endif //COMPONENTS_HPP
