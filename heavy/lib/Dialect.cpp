//===--- Dialect.cpp - HeavyScheme MLIR Dialect Registration --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for HeavyScheme's MLIR Dialect
//
//===----------------------------------------------------------------------===//

#include "heavy/Dialect.h"

#include "mlir/IR/Builder.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace heavy;

Dialect::Dialect(mlir::MLIRContext* Ctx) : mlir::Dialect("heavy", Ctx) {
  addOperations<
#define GET_OP_LIST
#include "heavy/Ops.cpp.inc"
    >();
}
