cmake_minimum_required(VERSION 3.13.4)

add_llvm_library(HeavyClang MODULE HeavyClang.cpp PLUGIN_TOOL clang)

# Access clang generated includes.
target_include_directories(HeavyClang
  PRIVATE ${CMAKE_BINARY_DIR}/tools/clang/include)
target_link_libraries(HeavyClang PRIVATE heavy)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(HeavyClang PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangLex
    )
endif(WIN32 OR CYGWIN)
