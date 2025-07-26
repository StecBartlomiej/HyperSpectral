

add_library(hyper_ml STATIC classification.cu)

target_include_directories(hyper_ml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(hyper_ml PRIVATE
        hyper_utility
        cuda_interface
        cublas
        cusolver
)
