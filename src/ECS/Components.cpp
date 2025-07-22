#include "Components.hpp"

#include "EntityComponentSystem.hpp"
#include "EnviHeader.hpp"


extern Coordinator coordinator;


std::size_t FlattenIdx(ImageSize img, std::size_t channel, std::size_t height, std::size_t width)
{
    return channel * (img.width * img.height) + height * img.width + width;
}

bool operator==(const ImageSize &lhs, const ImageSize &rhs) noexcept
{
    return lhs.width == rhs.width && lhs.height == rhs.height && lhs.channel == rhs.channel;
}

void RegisterComponents()
{
    coordinator.RegisterComponent<FilesystemPaths>();
    coordinator.RegisterComponent<ImageSize>();
    coordinator.RegisterComponent<EnviHeader>();
    coordinator.RegisterComponent<PatchData>();
    LOG_INFO("Registered components");
}
