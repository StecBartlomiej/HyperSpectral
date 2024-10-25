#include "Components.hpp"

#include "EntityComponentSystem.hpp"
#include "EnviHeader.hpp"


extern Coordinator coordinator;


void RegisterComponents()
{
    coordinator.RegisterComponent<FilesystemPaths>();
    coordinator.RegisterComponent<ImageSize>();
    coordinator.RegisterComponent<EnviHeader>();
    LOG_INFO("Registered components");
}
