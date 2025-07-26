
set(SOURCE_FILES
        EnviHeader.cpp
        EnviLexer.cpp
        EnviParser.cpp
)

add_library(hyper_parser STATIC ${SOURCE_FILES})

target_include_directories(hyper_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hyper_parser PUBLIC
        hyper_utility
)
