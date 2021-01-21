cmake_minimum_required(VERSION 3.4.3)

add_llvm_executable(heavy-scheme heavy_main.cpp
             ${HEAVY_LIB}/OpGen.cpp
             ${HEAVY_LIB}/HeavyScheme.cpp
             ${HEAVY_LIB}/Lexer.cpp
             ${HEAVY_LIB}/Parser.cpp
             ${HEAVY_LIB}/../../clang/lib/Basic/CharInfo.cpp)

llvm_map_components_to_libnames(LLVM_LIBS
  core
  support)

set(MLIR_LIBS
  MLIRAnalysis
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTransforms)

target_link_libraries(heavy-scheme PRIVATE ${LLVM_LIBS} ${MLIR_LIBS})
            
