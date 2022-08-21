cmake_minimum_required(VERSION 3.16)
project(NN LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS " -W -Wextra -std=c++17 -g3 -ggdb3 -fsanitize=address,undefined -D_GLIBCXX_DEBUG")

add_executable(main main.cu nn.hpp nn.h)