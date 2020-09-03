cmake_minimum_required(VERSION 3.5.1)


# ------------------------- Components -------------------------
# common-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)

# decoder-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/decoder)

# lm-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lm)

# ------------------------- Library -------------------------

add_library(
  ctcdecoder-libraries
  ""
  )

set_target_properties(
  ctcdecoder-libraries
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  ctcdecoder-libraries
  PUBLIC
  common-library
  decoder-library
  lm-library
  )

target_include_directories(
  ctcdecoder-libraries
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  ctcdecoder-libraries
  PUBLIC
  ${W2L_COMPILE_DEFINITIONS}
  )
