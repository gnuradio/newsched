

add_executable(test_cuda2 test_cuda.cpp)

target_include_directories(gnuradio-scheduler-cuda
PUBLIC
$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
PRIVATE
${PROJECT_SOURCE_DIR}/schedulers/simplestream/include
${PROJECT_SOURCE_DIR}/schedulers/cuda/include
${PROJECT_SOURCE_DIR}/runtime/include 
)


target_link_libraries(test_cuda2
    gnuradio-runtime
    gnuradio-blocklib-blocks
    gnuradio-blocklib-cuda-cu
    gnuradio-blocklib-cuda
    gnuradio-scheduler-simplestream
    gnuradio-scheduler-cuda
)

