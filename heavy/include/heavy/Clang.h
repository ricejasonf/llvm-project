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
#define HEAVY_CLANG_LIB_STR           "_HEAVYL5SheavyL5Sclang"
#define HEAVY_CLANG_LOAD_MODULE       HEAVY_CLANG_LIB##_load_module
#define HEAVY_CLANG_INIT              HEAVY_CLANG_LIB##_init
#define HEAVY_CLANG_VAR(NAME)         HEAVY_CLANG_VAR__##NAME
#define HEAVY_CLANG_VAR__diag_error   HEAVY_CLANG_LIB_(VS4diagmi5Serror)
#define HEAVY_CLANG_VAR__hello_world  HEAVY_CLANG_LIB_(V5Shellomi5Sworld)
#define HEAVY_CLANG_VAR__write_lexer  HEAVY_CLANG_LIB_(V5Swritemi5Slexer)

// diag-error
extern heavy::ContextLocal HEAVY_CLANG_VAR(diag_error);

// hello-world
extern heavy::ContextLocal HEAVY_CLANG_VAR(hello_world);

// write-lexer
extern heavy::ContextLocal HEAVY_CLANG_VAR(write_lexer);

// expr-eval
extern heavy::ContextLocal HEAVY_CLANG_VAR(expr_eval);

extern "C" {
// initialize the module for run-time independent of the compiler
inline void HEAVY_CLANG_INIT(heavy::Context& Context) {
  assert(HEAVY_CLANG_VAR(diag_error).get(Context) &&
      "external module must be preloaded");
  assert(HEAVY_CLANG_VAR(hello_world).get(Context) &&
      "external module must be preloaded");
  assert(HEAVY_CLANG_VAR(write_lexer).get(Context) &&
      "external module must be preloaded");
  assert(HEAVY_CLANG_VAR(expr_eval).get(Context) &&
      "external module must be preloaded");
}

// initializes the module and loads lookup information
// for the compiler
inline void HEAVY_CLANG_LOAD_MODULE(heavy::Context& Context) {
  HEAVY_CLANG_INIT(Context);
  heavy::initModule(Context, HEAVY_CLANG_LIB_STR, {
    {"diag-error",  HEAVY_CLANG_VAR(diag_error).init(Context)},
    {"hello-world", HEAVY_CLANG_VAR(hello_world).init(Context)},
    {"write-lexer", HEAVY_CLANG_VAR(write_lexer).init(Context)},
    {"expr-eval",   HEAVY_CLANG_VAR(expr_eval).init(Context)}
  });
}
}

#endif
