cmake_minimum_required(VERSION 3.4.3)
include(AddLLVM)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

set(HEAVY_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CLANG_LIB ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/lib)
include_directories(BEFORE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include
                    ${CMAKE_BINARY_DIR}/tools/clang/include)
add_subdirectory(tools/heavy)
