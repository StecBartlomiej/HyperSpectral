
set(SOURCE_FILES
        MainWindow.cpp
)

add_library(hyper_gui STATIC ${SOURCE_FILES})

target_include_directories(hyper_gui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hyper_gui PRIVATE
        cereal_lib
        Qt::Core
        Qt::Widgets
        hyper_utility
        hyper_ecs
        hyper_image
)

set_target_properties(hyper_gui PROPERTIES
        WIN32_EXECUTABLE ON
        MACOSX_BUNDLE ON
)
