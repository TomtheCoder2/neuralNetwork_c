cmake_minimum_required(VERSION 3.16)
project(NN LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS " -W -Wextra -std=c++17 -pg -O3 -g3 -ggdb3 -fsanitize=address,undefined -D_GLIBCXX_DEBUG")

add_executable(main main.cu nn.hpp nn.h stacktrace.h)