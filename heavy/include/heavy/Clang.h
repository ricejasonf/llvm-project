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

#define HEAVY_CLANG_LIB               _HEAVYL5clang
#define HEAVY_CLANG_LIB_STR          "_HEAVYL5clang"
#define HEAVY_CLANG_IS_LOADED         HEAVY_CLANG_LIB##_is_loaded
#define HEAVY_CLANG_LOAD_MODULE       HEAVY_CLANG_LIB##_LOAD_MODULE
#define HEAVY_CLANG_INIT              HEAVY_CLANG_LIB##_init
#define HEAVY_CLANG_VAR(NAME)         HEAVY_CLANG_VAR__##NAME
#define HEAVY_CLANG_VAR__diag_error   _HEAVYL5clangV4diagmi5error
#define HEAVY_CLANG_VAR__hello_world  _HEAVYL5clangV5hellomi5world

static bool HEAVY_CLANG_IS_LOADED = false;

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
    heavy::createModule(Context, HEAVY_CLANG_LIB_STR, {
      {"diag-error",  HEAVY_CLANG_VAR(diag_error)},
      {"hello-world", HEAVY_CLANG_VAR(hello_world)}
    });
  }
}

#endif
