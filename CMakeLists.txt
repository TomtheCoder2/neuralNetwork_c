cmake_minimum_required(VERSION 3.16)
project(Neural_Network C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS " -W -Wextra -std=c++17 -g3 -ggdb3 -fsanitize=address,undefined -D_GLIBCXX_DEBUG -lm -O3 -finline-functions -funroll-loops -I $(AMDAPPSDKROOT)/include -L $(AMDAPPSDKROOT)/lib -lOpenCL")
#target_link_libraries(train m)
add_executable(train_p nn_photemy.c)
add_executable(train train.c)
add_executable(plotTest plotTest.c)
add_executable(openCLTest openCLTest.c)

target_link_libraries(train m)
target_link_libraries(plotTest m)
target_link_libraries(train_p m)