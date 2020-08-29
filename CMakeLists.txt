cmake_minimum_required(VERSION 3.5.1)

project(ctcdecoder++)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# ----------------------------- Configuration -----------------------------

option(W2L_BUILD_LIBRARIES_ONLY "Build ctcdecoder-libraries only" OFF)
option(W2L_LIBRARIES_USE_CUDA "Use CUDA in libraries-only build" ON)
option(W2L_LIBRARIES_USE_KENLM "Use KenLM in libraries-only build" ON)
option(W2L_LIBRARIES_USE_MKL "Use MKL in libraries-only build" OFF)
option(W2L_BUILD_FOR_PYTHON "Build Python bindings" OFF)
set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

# ------------------------- Dependency Fallback -------------------------

if (W2L_LIBRARIES_USE_MKL)
  find_package(MKL)
  if (NOT MKL_FOUND)
    message(WARNING "MKL not found; forcing W2L_LIBRARIES_USE_MKL=OFF.")
    set(W2L_LIBRARIES_USE_MKL OFF CACHE BOOL "" FORCE)
  endif ()
endif ()

# ------------------------- Libraries-Only Build ------------------------------

function (build_w2l_libraries)
  set(W2L_COMPILE_DEFINITIONS
    $<$<BOOL:${W2L_LIBRARIES_USE_CUDA}>:W2L_LIBRARIES_USE_CUDA>
    $<$<BOOL:${W2L_LIBRARIES_USE_KENLM}>:W2L_LIBRARIES_USE_KENLM>
    $<$<BOOL:${W2L_LIBRARIES_USE_MKL}>:W2L_LIBRARIES_USE_MKL>
    )
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
    
    if (W2L_BUILD_FOR_PYTHON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/bindings/python)
  endif ()
endfunction ()

if (W2L_BUILD_LIBRARIES_ONLY)
  build_w2l_libraries()
  return()
endif ()
# ------------------------- Components -------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ------------------------- Library -------------------------

add_library(
  ctcdecoder++
  ""
  )

set_target_properties(
  ctcdecoder++
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  ctcdecoder++
  PUBLIC
  ctcdecoder-libraries
  )

target_include_directories(
  ctcdecoder++
  PUBLIC
  )

