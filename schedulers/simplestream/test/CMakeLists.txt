

add_executable(test_parameters_callbacks test_parameters_callbacks.cpp)
add_executable(test_multiple_output test_multiple_output.cpp)
add_executable(test_one_to_many test_one_to_many.cpp)
add_executable(test_multisched test_multisched.cpp)

target_include_directories(gnuradio-scheduler-simplestream
PUBLIC
$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
PRIVATE
${PROJECT_SOURCE_DIR}/schedulers/simplestream/include
${PROJECT_SOURCE_DIR}/runtime/include 
)

target_link_libraries(test_parameters_callbacks 
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

target_link_libraries(test_multisched 
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream
)


list(APPEND TEST_SOURCES
qa_one_to_many.cpp
# qa_multiple_output.cpp
)

add_executable(qa_simplestream ${TEST_SOURCES})
target_include_directories(qa_simplestream
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
    # PRIVATE
    # ${CMAKE_SOURCE_DIR}/deps/catch/include
    
)
target_link_libraries(qa_simplestream  
    Catch2::Catch2   
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream)
add_test(NAME test
    COMMAND qa_simplestream)