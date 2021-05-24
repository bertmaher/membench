cmake_minimum_required(VERSION 3.10)

project(membench)

find_package(benchmark REQUIRED)
add_executable(mem_bench mem_bench.cpp)
target_compile_options(mem_bench PUBLIC -march=native)
set_target_properties(mem_bench PROPERTIES CXX_STANDARD 17)
target_link_libraries(mem_bench benchmark::benchmark)
