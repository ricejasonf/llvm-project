cmake_minimum_required(VERSION 3.4.3)

add_llvm_library(heavy
  Builtins.cpp
  Context.cpp
  Dialect.cpp
  Heap.cpp
  HeavyScheme.cpp
  Lexer.cpp
  Mangle.cpp
  OpEval.cpp
  OpGen.cpp
  Parser.cpp
  Quasiquote.cpp
  SourceFileStorage.cpp
  SourceManager.cpp
  # TODO Do not use Clangs CharInfo.
  ${CLANG_LIB_DIR}/Basic/CharInfo.cpp

  LINK_COMPONENTS
    Core
    Support

  LINK_LIBS
    MLIRAnalysis
    MLIRFuncDialect
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRTransforms
  )

add_dependencies(heavy HeavyOpsIncGen
                       MLIRFuncOpsIncGen)

#set(MLIR_LIBS
#  MLIRAnalysis
#  MLIRFuncDialect
#  MLIRIR
#  MLIRSideEffectInterfaces
#  MLIRTransforms)
#
#target_link_libraries(heavy PRIVATE ${LLVM_LIBS} ${MLIR_LIBS})
