

add_executable(test_pmt test_pmt.cpp)

target_include_directories(test_pmt
PUBLIC
$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(test_pmt 
    pmt
)


########################################################
#   QA Tests
########################################################

list(APPEND TEST_SOURCES
qa_pmt.cpp
)

add_executable(qa_pmt ${TEST_SOURCES})
target_include_directories(qa_pmt
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_SOURCE_DIR}/deps/catch/include
    
)
target_link_libraries(qa_pmt  
    pmt)

add_test(NAME pmt_tests
    COMMAND qa_pmt)
