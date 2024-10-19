
set(SOURCE_FILES
        Logger.cpp
        EnviHeader.cpp
        EnviLexer.cpp
        EnviParser.cpp
        EntityComponentSystem.cpp
)

add_library(lib_hypercpp STATIC ${SOURCE_FILES})
target_include_directories(lib_hypercpp PUBLIC "."  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(lib_hypercpp PUBLIC spdlog::spdlog)
set_target_properties(lib_hypercpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
