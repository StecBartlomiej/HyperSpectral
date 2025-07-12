
set(SOURCE_FILES
        Logger.cpp
        Parser/EnviHeader.cpp
        Parser/EnviLexer.cpp
        Parser/EnviParser.cpp
        ECS/EntityComponentSystem.cpp
        ECS/Components.cpp
        Image.cu
        Classification.cu
)

set(GUI_SOURCE_FILES
        Gui.cpp
)


add_library(lib_hypercpp STATIC ${SOURCE_FILES})
target_include_directories(lib_hypercpp PUBLIC "." "Parser" "ECS")
target_link_libraries(lib_hypercpp PUBLIC
        spdlog::spdlog
        cuda_interface
        cublas
        cusolver
        cereal_lib
        )
target_compile_definitions(lib_hypercpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:COMPILING_CXX>:/arch:AVX512 /NODEFAULTLIB:LIBCMT /MT /O2 /GL /LTCG
)


add_library(lib_gui STATIC ${GUI_SOURCE_FILES})
target_link_libraries(lib_gui PUBLIC
        glad_gl_core_43
        glfw
        imgui
        implot
        imgui_node
        lib_hypercpp
)
target_compile_options(lib_gui PUBLIC /utf-8)
