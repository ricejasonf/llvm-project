//===----- Clang.h - HeavyScheme module for clang bindings ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains an interface to initialize a Scheme module from Clang.
//  The names use HeavyScheme's mangling to allow for a consistent interface
//  for loading modules from precompiled code.
//  Ideally files like this should be generated, but this serves as a reference.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_CLANG_H
#define LLVM_HEAVY_CLANG_H

#include "heavy/Value.h"

#define HEAVY_CLANG_LIB               _HEAVYL5Sclang
#define HEAVY_CLANG_LIB_(NAME)        _HEAVYL5Sclang ## NAME
#define HEAVY_CLANG_LIB_STR          "_HEAVYL5Sclang"
#define HEAVY_CLANG_IS_LOADED         HEAVY_CLANG_LIB##_is_loaded
#define HEAVY_CLANG_LOAD_MODULE       HEAVY_CLANG_LIB##_load_module
#define HEAVY_CLANG_INIT              HEAVY_CLANG_LIB##_init
#define HEAVY_CLANG_VAR(NAME)         HEAVY_CLANG_VAR__##NAME
#define HEAVY_CLANG_VAR__diag_error   HEAVY_CLANG_LIB_(VS4diagmi5Serror)
#define HEAVY_CLANG_VAR__hello_world  HEAVY_CLANG_LIB_(V5Shellomi5Sworld)
// #define HEAVY_CLANG_VAR__diag_error   _HEAVYL5clangVS4diagmi5Serror
// #define HEAVY_CLANG_VAR__hello_world  _HEAVYL5clangV5Shellomi5Sworld

extern bool HEAVY_CLANG_IS_LOADED;

// diag-error
extern heavy::ExternLambda<1> HEAVY_CLANG_VAR(diag_error);

// hello-world
extern heavy::ExternLambda<1> HEAVY_CLANG_VAR(hello_world);

extern "C" {
// initialize the module for run-time independent of the compiler
inline void HEAVY_CLANG_INIT(heavy::Context& Context) {
  assert(!HEAVY_CLANG_IS_LOADED &&
    "module should not be loaded more than once");
  HEAVY_CLANG_IS_LOADED = true;
  assert(HEAVY_CLANG_VAR(diag_error).Value &&
      "external module must be preloaded");
  assert(HEAVY_CLANG_VAR(hello_world).Value &&
      "external module must be preloaded");
}

// initializes the module and loads lookup information
// for the compiler
inline void HEAVY_CLANG_LOAD_MODULE(heavy::Context& Context) {
  HEAVY_CLANG_INIT(Context);
  heavy::initModule(Context, HEAVY_CLANG_LIB_STR, {
    {"diag-error",  HEAVY_CLANG_VAR(diag_error)},
    {"hello-world", HEAVY_CLANG_VAR(hello_world)}
  });
}
}

#endif
