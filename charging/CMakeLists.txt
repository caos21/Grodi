cmake_minimum_required(VERSION 3.0)
project(charging)

# ============= LSODA
set(LSODA_DIR "${PROJECT_SOURCE_DIR}/external/liblsoda")
message(STATUS "lsoda dir: ${LSODA_DIR}")

include(ExternalProject)
ExternalProject_Add(liblsoda
    SOURCE_DIR "${LSODA_DIR}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

include_directories(SYSTEM "${LSODA_DIR}/src")

# 1. Change STATIC to SHARED here
add_library(lsoda SHARED IMPORTED GLOBAL)
add_dependencies(lsoda liblsoda)

# 2. Update the filename extension to .so
set_target_properties(lsoda PROPERTIES
    IMPORTED_LOCATION "${LSODA_DIR}/src/liblsoda.so"
)
# ============= LSODA


include_directories("${PROJECT_SOURCE_DIR}/include")

# This will now successfully find src/charging.c relative to this file
add_library(charging SHARED src/charging.c include/charging.h)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Ofast -fPIC -Wall -march=znver1")
set(CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} -fopenmp -Ofast -Wall -march=znver1")

find_package(OpenMP)
if(OPENMP_FOUND)
  message("** OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ")
  add_definitions(-DOPENMP)
endif()

target_link_libraries(charging lsoda -lm)

enable_testing()

# Install the final compiled .so into the exact 'lib' folder setup.py expects
install(TARGETS charging DESTINATION "${PROJECT_SOURCE_DIR}/lib")
