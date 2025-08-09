
set(SOURCE_FILES
        MainWindow.cpp
        ViewImageWidget.cpp
        view_image.ui
)

add_library(hyper_gui STATIC ${SOURCE_FILES})

target_include_directories(hyper_gui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hyper_gui PUBLIC
        Qt6::Core
        Qt6::Widgets
        Qt6::Gui
        Qt6::OpenGLWidgets
        Qt6::Charts
        hyper_utility
)


target_link_libraries(hyper_gui PRIVATE
        cereal_lib
        hyper_ecs
        hyper_image
)

set_target_properties(hyper_gui PROPERTIES
        WIN32_EXECUTABLE ON
        MACOSX_BUNDLE ON
)
