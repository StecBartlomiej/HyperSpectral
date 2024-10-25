
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


# ==========================================


FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG master
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only" )
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only" )
set(GLFW_TEST OFF CACHE BOOL "GLFW lib only" )

# ==========================================

FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui
        GIT_TAG docking
)

FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    message("Fetching imgui")
    FetchContent_Populate(imgui)

    add_library(imgui
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_demo.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)

    target_include_directories(imgui PUBLIC
            ${imgui_SOURCE_DIR}
            ${imgui_SOURCE_DIR}/backends
            ${glfw_SOURCE_DIR}/include)

    target_link_libraries(imgui PRIVATE glfw)

    # copy font
    configure_file(${imgui_SOURCE_DIR}/misc/fonts/Roboto-Medium.ttf  ${CMAKE_BINARY_DIR} COPYONLY)
endif ()

# ==========================================

FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG master
)
FetchContent_GetProperties(implot)
if(NOT implot_POPULATED)
    message("Fetching implot")
    FetchContent_Populate(implot)

    add_library(implot
            ${implot_SOURCE_DIR}/implot.cpp
            ${implot_SOURCE_DIR}/implot_items.cpp
            ${implot_SOURCE_DIR}/implot_demo.cpp
    )

    target_include_directories(implot PUBLIC
            ${implot_SOURCE_DIR}
    )

    target_link_libraries(implot PRIVATE imgui)
endif ()

# ==========================================

FetchContent_MakeAvailable(glfw)

# ==========================================
include(FetchContent)

FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad
        GIT_TAG        v2.0.8
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  cmake
)

FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    message("Fetching glad")
    FetchContent_MakeAvailable(glad)

#    add_subdirectory("${glad_SOURCE_DIR}/cmake" glad_cmake)
#    glad_add_library(glad REPRODUCIBLE EXCLUDE_FROM_ALL LOADER API gl:core=3.3 EXTENSIONS GL_ARB_bindless_texture GL_EXT_texture_compression_s3tc)
    glad_add_library(glad_gl_core_43 STATIC REPRODUCIBLE LOADER API gl:core=3.3)
endif()
