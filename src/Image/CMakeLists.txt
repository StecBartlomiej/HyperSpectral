
set(SOURCE_FILES
        Image.cu
)

add_library(hyper_image STATIC ${SOURCE_FILES})

target_include_directories(hyper_image PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hyper_image PRIVATE
        cereal_lib
        hyper_utility
        hyper_ecs
        hyper_parser
        cuda_interface
        cublas
        cusolver
)
