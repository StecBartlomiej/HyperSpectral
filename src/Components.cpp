#include "Components.hpp"

#include "EntityComponentSystem.hpp"
#include "EnviHeader.hpp"


extern Coordinator coordinator;


bool operator==(const ImageSize &lhs, const ImageSize &rhs) noexcept
{
    return lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth;
}

void RegisterComponents()
{
    coordinator.RegisterComponent<FilesystemPaths>();
    coordinator.RegisterComponent<ImageSize>();
    coordinator.RegisterComponent<EnviHeader>();
    coordinator.RegisterComponent<TreeAttributes>();
    LOG_INFO("Registered components");
}
