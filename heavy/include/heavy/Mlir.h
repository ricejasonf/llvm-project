//===- Mlir.h - Mlir binding functions for HeavyScheme ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares values and functions for the (heavy mlir) scheme library
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_MLIR_H
#define LLVM_HEAVY_MLIR_H

#define HEAVY_MLIR_LIB                _HEAVYL5SheavyL4SmlirL8Sbuiltins
#define HEAVY_MLIR_LIB_(NAME)         _HEAVYL5SheavyL4SmlirL8Sbuiltins ## NAME
#define HEAVY_MLIR_LIB_STR            "_HEAVYL5SheavyL4SmlirL8Sbuiltins"
#define HEAVY_MLIR_LOAD_MODULE        HEAVY_MLIR_LIB_(_load_module)
#define HEAVY_MLIR_INIT               HEAVY_MLIR_LIB_(_init)

namespace heavy {
class Context;
}

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_MLIR_LOAD_MODULE(heavy::Context& Context);
void HEAVY_MLIR_INIT(heavy::Context& Context);
}

#endif  // LLVM_HEAVY_MLIR_H
