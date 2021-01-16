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

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Interfaces/SideEffectInterfaces.h"

namespace heavy {
class Dialect : public mlir::Dialect {
  explicit Dialect(mlir::MLIRContext* Ctx);
  static llvm::StringRef getDialectNamespace() { return "heavy"; }
};

class HeavyValue : public mlir::Type::TypeBase<
                            HeavyValue,
                            mlir::Type,
                            mlir::TypeStorage> {
  using Base::Base;
};

class HeavyValueAttr : public mlir::Type::AttributeBase<
                            HeavyValueAttr,
                            mlir::Attribute,
                            mlir::AttributeStorage> {
  using Base::Base;
};

#define GET_OP_CLASSES
#include "heavy/Ops.h.inc"
}

#endif
