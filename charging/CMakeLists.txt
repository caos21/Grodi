cmake_minimum_required(VERSION 3.0)
project(charging)


# ============= LSODA
set(LSODA_DIR "${PROJECT_SOURCE_DIR}/external/liblsoda/")
message("lsoda dir ${LSODA_DIR}")
include(ExternalProject)
ExternalProject_Add(liblsoda
    SOURCE_DIR ${LSODA_DIR}
    CONFIGURE_COMMAND ""
    STEP_TARGETS build
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND "")
include_directories("${LSODA_DIR}/src/")
link_directories("${LSODA_DIR}/src/")
# ============= LSODA


link_directories("${PROJECT_SOURCE_DIR}/lib")

include_directories("${PROJECT_SOURCE_DIR}/include")

add_library(charging SHARED src/charging.c include/charging.h)

set(LIBS "-lm")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Ofast -fPIC -Wall -ftree-vectorizer-verbose=3 -march=znver1 -ftree-vectorize -fopt-info-vec-missed=vec.miss -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security")
set(CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} -fopenmp -Ofast -Wall -ftree-vectorizer-verbose=3 -march=znver1 -ftree-vectorize -fopt-info-vec-missed=vec.miss -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security")

find_package(OpenMP)
if(OPENMP_FOUND)
  message("** OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ")
  add_definitions(-DOPENMP)
endif()

target_link_libraries(charging
                      lsoda
                      ${LIBS})

enable_testing()
install(TARGETS charging DESTINATION "${PROJECT_SOURCE_DIR}/lib")
