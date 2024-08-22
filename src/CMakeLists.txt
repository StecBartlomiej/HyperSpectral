
set(SOURCE_FILES
        ImageHeader.cpp
        Logger.cpp
        EnviLexer.cpp
)

add_library(lib_hypercpp STATIC ${SOURCE_FILES})
target_include_directories(lib_hypercpp PUBLIC ".")
target_link_libraries(lib_hypercpp PUBLIC spdlog::spdlog)
