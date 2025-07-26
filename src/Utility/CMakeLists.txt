add_library(hyper_utility STATIC Logger.cpp)

target_include_directories(hyper_utility PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hyper_utility PUBLIC
        spdlog::spdlog
        cereal_lib
)
