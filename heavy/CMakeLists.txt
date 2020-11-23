cmake_minimum_required(VERSION 3.4.3)
include(AddLLVM)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

add_subdirectory(heavy)
