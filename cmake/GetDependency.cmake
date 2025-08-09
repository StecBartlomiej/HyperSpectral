
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# ==========================================

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.14.1
)
FetchContent_MakeAvailable(spdlog)

# ============================================

FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal
        GIT_TAG        v1.3.2
)

set(BUILD_DOC OFF CACHE BOOL "Disable documentation for cereal" FORCE)
set(BUILD_SANDBOX OFF CACHE BOOL "Disable examples for cereal" FORCE)
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Skip performance comparison for cereal" FORCE)

FetchContent_MakeAvailable(cereal)

add_library(cereal_lib INTERFACE)
target_include_directories(cereal_lib INTERFACE "${cereal_SOURCE_DIR}/include/")

# ==========================================

add_library(cuda_interface INTERFACE)
target_include_directories(cuda_interface INTERFACE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
set_target_properties(cuda_interface PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cuda_interface INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)

# ==========================================
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui OpenGLWidgets Charts)
qt_standard_project_setup()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
