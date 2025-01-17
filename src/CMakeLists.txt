
set(SOURCE_FILES
        Logger.cpp
        EnviHeader.cpp
        EnviLexer.cpp
        EnviParser.cpp
        EntityComponentSystem.cpp
        Components.cpp
        Image.cu
        Classification.cu
)

set(GUI_SOURCE_FILES
        Gui.cpp
)


add_library(lib_hypercpp STATIC ${SOURCE_FILES})
target_include_directories(lib_hypercpp PUBLIC "."  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "${cereal_SOURCE_DIR}/include/" "${imgui_node_SOURCE_DIR}")
target_link_libraries(lib_hypercpp PUBLIC spdlog::spdlog cublas cusolver)
set_target_properties(lib_hypercpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(lib_hypercpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)
target_compile_definitions(lib_hypercpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:COMPILING_CXX>:/arch:AVX512 /NODEFAULTLIB:LIBCMT /MTd /O2 /GL /LTCG)


add_library(lib_gui STATIC ${GUI_SOURCE_FILES})
target_link_libraries(lib_gui PUBLIC glad_gl_core_43)
target_link_libraries(lib_gui PUBLIC glfw imgui implot imgui_node lib_hypercpp)
target_compile_options(lib_gui PUBLIC /utf-8)
