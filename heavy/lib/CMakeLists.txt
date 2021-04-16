cmake_minimum_required(VERSION 3.4.3)

add_llvm_library(heavy
  ${HEAVY_LIB}/Builtins.cpp
  ${HEAVY_LIB}/Context.cpp
  ${HEAVY_LIB}/Dialect.cpp
  ${HEAVY_LIB}/OpEval.cpp
  ${HEAVY_LIB}/OpGen.cpp
  ${HEAVY_LIB}/Lexer.cpp
  ${HEAVY_LIB}/Parser.cpp
  ${HEAVY_LIB}/Quasiquote.cpp
  ${HEAVY_LIB}/../../clang/lib/Basic/CharInfo.cpp)

add_dependencies(heavy HeavyOpsIncGen)

llvm_map_components_to_libnames(LLVM_LIBS
  core
  support)

set(MLIR_LIBS
  MLIRAnalysis
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTransforms)

target_link_libraries(heavy PRIVATE ${LLVM_LIBS} ${MLIR_LIBS})
