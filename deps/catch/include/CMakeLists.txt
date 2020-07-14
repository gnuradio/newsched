########################################################################
# Install header files
########################################################################
install(FILES
  catch.hpp
  DESTINATION ${GR_INCLUDE_DIR}
)

message(STATUS ${CMAKE_INSTALL_INCLUDEDIR})