cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

add_llvm_executable(heavy heavy_main.cpp
                          HeavyScheme.cpp)
