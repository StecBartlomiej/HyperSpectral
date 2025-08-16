#include "Components.hpp"
#include "MainWindow.hpp"
#include "Logger.hpp"
#include "EntityComponentSystem.hpp"


Coordinator coordinator{};

int main(int argc, char *argv[])
{
    Logger::Init(spdlog::level::trace);
    RegisterComponents();
    return Run(argc, argv);
}
