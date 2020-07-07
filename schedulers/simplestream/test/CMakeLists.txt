

add_executable(test_simplestream test_simplestream.cpp)
add_executable(test_multiple_output test_multiple_output.cpp)
add_executable(test_one_to_many test_one_to_many.cpp)

target_include_directories(gnuradio-scheduler-simplestream
PUBLIC
$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
PRIVATE
${PROJECT_SOURCE_DIR}/schedulers/simplestream/include
${PROJECT_SOURCE_DIR}/runtime/include 
)

target_link_libraries(test_simplestream 
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream
)

target_link_libraries(test_multiple_output 
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream
)

target_link_libraries(test_one_to_many 
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream
)