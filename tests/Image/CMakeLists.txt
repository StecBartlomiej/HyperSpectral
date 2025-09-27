set(SOURCE_FILES
        ../main.cpp
        ImageDataBufferTest.cu

)

add_executable(test_hyper_image ${SOURCE_FILES})

target_link_libraries(test_hyper_image PUBLIC
        hyper_image
        Catch2::Catch2
)
catch_discover_tests(test_hyper_image)
