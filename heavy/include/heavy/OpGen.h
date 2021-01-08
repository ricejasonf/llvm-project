//===------ OpGen.h - Classes for generating MLIR Operations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::OpGen for syntax expansion and operation generation
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_OP_GEN_H
#define LLVM_HEAVY_OP_GEN_H

#include "heavy/HeavyScheme.h"
#include "llvm/Support/Casting.h"

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
  class Value;
}

namespace heavy {

mlir::OwningModuleRef opGen(Context&, Value*);

}

namespace heavy { namespace builtin_syntax {

mlir::Value define(Context& C, Pair* P);
mlir::Value quote(Context& C, Pair* P);
mlir::Value quasiquote(Context& C, Pair* P);

}}

#endif
