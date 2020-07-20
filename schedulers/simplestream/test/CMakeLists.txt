

add_executable(test_parameters_callbacks test_parameters_callbacks.cpp)
add_executable(test_one_to_many test_one_to_many.cpp)
add_executable(test_multisched test_multisched.cpp)
add_executable(test_multi_da test_multi_da.cpp)

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

target_link_libraries(test_multi_da
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream
)

########################################################
#   QA Tests
########################################################

list(APPEND TEST_SOURCES
qa_scheduler_simplestream.cpp
)

add_executable(qa_simplestream ${TEST_SOURCES})
target_include_directories(qa_simplestream
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_SOURCE_DIR}/deps/catch/include
    
)
target_link_libraries(qa_simplestream  
    gnuradio-blocklib-common
    gnuradio-blocklib-blocks
    gnuradio-runtime
    gnuradio-scheduler-simplestream)

add_test(NAME test
    COMMAND qa_simplestream)
