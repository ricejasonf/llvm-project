//===- Dialect.h - Classes for representing declarations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Declares the mlir Dialect including generated classes
//
//===----------------------------------------------------------------------===//

#ifndef HEAVY_DIALECT_H
#define HEAVY_DIALECT_H

#include "heavy/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

// In Value we assume Operation* has the same alignment as ValueBase*.
// (which should be 8 bytes)
// Check that here
static_assert(llvm::PointerLikeTypeTraits<heavy::Operation*>::NumLowBitsAvailable ==
              heavy::ValueSumType::OperationTraits::NumLowBitsAvailable,
              "mlir::Operation* must have 8 byte alignment to fit in heavy::Value");


namespace heavy {
using mlir::func::FuncOp;
using StringAttr = mlir::StringAttr;
}

namespace heavy::detail {
struct HeavyValueAttrStorage;
}

// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wunused-parameter"

#include "heavy/Dialect/HeavyDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "heavy/Dialect/HeavyTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "heavy/Dialect/HeavyAttrs.h.inc"

#define GET_OP_CLASSES
#include "heavy/Dialect/HeavyOps.h.inc"
// #pragma clang diagnostic pop

#endif
