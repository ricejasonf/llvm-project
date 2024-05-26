cmake_minimum_required(VERSION 3.13.4)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddHeavy)

#set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

set(HEAVY_GEN_INC ${CMAKE_BINARY_DIR}/tools/heavy/include)
set(CLANG_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../clang/lib)
include_directories(BEFORE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include
                    ${CMAKE_BINARY_DIR}/tools/mlir/include
                    ${HEAVY_GEN_INC}
                    ../clang/include)

add_subdirectory(include/heavy)
add_subdirectory(lib)
add_subdirectory(tools/heavy-scheme)
add_subdirectory(test)
add_subdirectory(unittests)

add_subdirectory(cmake/modules)
