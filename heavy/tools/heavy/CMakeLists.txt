cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

add_llvm_executable(heavy heavy_main.cpp
             ${HEAVY_LIB}/HeavyScheme.cpp
             ${HEAVY_LIB}/Lexer.cpp
             ${HEAVY_LIB}/Parser.cpp)

target_link_libraries(heavy
  PRIVATE
  clangBasic
  ${LLVM_COMMON_LIBS})

# ClangDiagnosticCommon for some *.inc file
add_dependencies(heavy ClangDiagnosticCommon)
            
