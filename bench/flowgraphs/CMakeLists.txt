add_executable(bm_basic basic.cpp)
add_executable(bm_domain_adapt domain_adapt.cpp)

target_include_directories(bm_basic
PUBLIC
$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
PRIVATE
${PROJECT_SOURCE_DIR}/schedulers/simplestream/include
${PROJECT_SOURCE_DIR}/runtime/include 
)

target_include_directories(bm_domain_adapt
PUBLIC
$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
PRIVATE
${PROJECT_SOURCE_DIR}/schedulers/simplestream/include
${PROJECT_SOURCE_DIR}/runtime/include 
)


target_link_libraries(bm_basic 
    gnuradio-runtime
    gnuradio-blocklib-blocks
    gnuradio-scheduler-simplestream
    Boost::program_options
)

target_link_libraries(bm_domain_adapt 
    gnuradio-runtime
    gnuradio-blocklib-blocks
    gnuradio-scheduler-simplestream
    Boost::program_options
)
