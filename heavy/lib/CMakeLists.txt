cmake_minimum_required(VERSION 3.4.3)

add_heavy_library(heavy
  Builtins.cpp
  Context.cpp
  Dialect.cpp
  Heap.cpp
  HeavyScheme.cpp
  Lexer.cpp
  Mangle.cpp
  Mlir.cpp
  OpEval.cpp
  OpGen.cpp
  Parser.cpp
  Quasiquote.cpp
  SourceFileStorage.cpp
  SourceManager.cpp
  NbdlDialect.cpp
  # TODO Do not use Clangs CharInfo.
  ${CLANG_LIB_DIR}/Basic/CharInfo.cpp

  LINK_COMPONENTS
    Core
    Support

  LINK_LIBS
    MLIRAnalysis
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms

  DEPENDS
    HeavyOpsIncGen
    NbdlOpsIncGen
    MLIRFuncOpsIncGen
  )
