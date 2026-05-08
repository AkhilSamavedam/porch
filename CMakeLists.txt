cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
    porch
    VERSION 0.1.0
    DESCRIPTION "A C++ tensor library with a CUDA JIT backend"
    LANGUAGES CXX
)

option(PORCH_BUILD_TESTS "Build porch tests" ON)

add_library(porch src/backend.cpp src/cuda_jit.cpp src/tensor.cpp)
add_library(porch::porch ALIAS porch)

target_compile_features(porch PUBLIC cxx_std_20)
target_compile_options(
    porch
    PUBLIC
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-std=c++20>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/std:c++20>
)
target_include_directories(
    porch
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(porch PRIVATE ${CMAKE_DL_LIBS})

if(MSVC)
    target_compile_options(porch PRIVATE /W4 /permissive-)
else()
    target_compile_options(porch PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(PORCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
