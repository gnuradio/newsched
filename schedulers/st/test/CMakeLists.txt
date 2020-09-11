

add_executable(test_scheduler_st test_scheduler_st.cpp)

target_link_libraries(test_scheduler_st 
    gnuradio-runtime
    gnuradio-blocklib-blocks
    gnuradio-scheduler-st
)


add_executable(st_cuda test_st_cuda.cpp)

target_link_libraries(st_cuda 
    gnuradio-runtime
    gnuradio-blocklib-blocks
    gnuradio-scheduler-st
    gnuradio-blocklib-cuda-cu
    gnuradio-blocklib-cuda
)    



########################################################
#   QA Tests
########################################################

# list(APPEND TEST_SOURCES
# qa_scheduler_default.cpp
# qa_tags.cpp
# )

# add_executable(qa_default ${TEST_SOURCES})
# target_include_directories(qa_default
#     PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
#     $<INSTALL_INTERFACE:include>
#     PRIVATE
#     ${CMAKE_SOURCE_DIR}/deps/catch/include
    
# )
# target_link_libraries(qa_default  
#     gnuradio-runtime
#     gnuradio-blocklib-blocks
#     gnuradio-scheduler-st)

# add_test(NAME default_tests
#     COMMAND qa_default)
