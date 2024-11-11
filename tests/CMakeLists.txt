
set(TEST_FILES
        main.cpp
        EnviLexerTest.cpp
        EnviParserTest.cpp
        EnviHeaderTest.cpp
        EntityComponentSystemTest.cpp
        ImageCudaTest.cu
)

add_executable(tests ${TEST_FILES})
target_link_libraries(tests PRIVATE lib_hypercpp Catch2::Catch2)
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(tests)
