cmake_minimum_required(VERSION 3.27)
project(HyperSpectral)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEBUG_INFO ON)
set(DISABLE_LOGGING OFF)


if (MSVC)
    add_compile_options(/W4)
else ()
    add_compile_options(-O3 -Wall -Werror -Wextra -Wshadow -Wpedantic)
endif ()

if (${DEBUG_INFO} AND UNIX)
    add_compile_options(-g)
    if (UNIX)
        add_compile_options(-fsanitize=address)
    endif ()
endif ()

enable_testing()
include(FetchContent)
include(GetDependency)


add_subdirectory(src)
add_subdirectory(tests)

add_executable(hypercpp main.cpp)
target_link_libraries(hypercpp PUBLIC lib_hypercpp)
