#include "Logger.hpp"

#include <catch2/catch_session.hpp>


int main(int argc, char* argv[])
{
    Logger::Init(spdlog::level::info);

    int result = Catch::Session().run( argc, argv );

    return result;
}