cmake_minimum_required(VERSION 3.5)

project(ctests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "/opt/libtorch/share/cmake/Torch")

find_package (Torch REQUIRED)

add_executable(ctests main.cpp)
target_link_libraries (ctests)
install(TARGETS ctests
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
