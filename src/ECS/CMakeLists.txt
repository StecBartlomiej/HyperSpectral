
set(SOURCE_FILES
        EntityComponentSystem.cpp
        Components.cpp
)

add_library(hyper_ecs STATIC ${SOURCE_FILES})

target_include_directories(hyper_ecs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hyper_ecs PRIVATE
        hyper_parser
)

target_link_libraries(hyper_ecs PUBLIC
        hyper_utility
)
