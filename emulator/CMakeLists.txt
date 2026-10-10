cmake_minimum_required(VERSION 3.16)
project(ff1d CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_ffcore
  src/bindings.cpp
  src/solver.cpp
  src/diffusion.cpp
  src/velocity.cpp
  src/advection.cpp
  src/reaction.cpp
  src/ignition.cpp
  src/qoi.cpp
)
set_target_properties(_ffcore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
