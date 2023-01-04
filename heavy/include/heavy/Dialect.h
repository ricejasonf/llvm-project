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
using mlir::FuncOp;

struct Dialect : public mlir::Dialect {
  explicit Dialect(mlir::MLIRContext* Ctx);
  static llvm::StringRef getDialectNamespace() { return "heavy"; }

  void printAttribute(mlir::Attribute Attr,
                      mlir::DialectAsmPrinter& P) const override;
  void printType(mlir::Type, mlir::DialectAsmPrinter&) const override;
};

struct HeavyValueTy : public mlir::Type::TypeBase<
                            HeavyValueTy,
                            mlir::Type,
                            mlir::TypeStorage> {
  using Base::Base;
};

struct HeavyRestTy : public mlir::Type::TypeBase<
                            HeavyRestTy,
                            mlir::Type,
                            mlir::TypeStorage> {
  using Base::Base;
};

struct HeavyValueAttrStorage : public mlir::AttributeStorage {
  heavy::Value Val;

  HeavyValueAttrStorage(heavy::Value V)
    : Val(V)
  { }

  using KeyTy = heavy::Value;
  bool operator==(KeyTy const& Key) const {
    return Key == Val;
  }

  static HeavyValueAttrStorage* construct(
      mlir::AttributeStorageAllocator& Allocator, heavy::Value V) {
    return new (Allocator.allocate<HeavyValueAttrStorage>())
      HeavyValueAttrStorage(V);
  }
};

class HeavyValueAttr : public mlir::Attribute::AttrBase<
                            HeavyValueAttr,
                            mlir::Attribute,
                            HeavyValueAttrStorage> {
  using Base::Base;

public:
  heavy::Value getValue() const;
};

// Additional Types
#define HEAVY_TYPE(NAME) \
struct Heavy##NAME##Ty : public mlir::Type::TypeBase< \
Heavy##NAME##Ty, mlir::Type, mlir::TypeStorage> { using Base::Base; }

HEAVY_TYPE(Pair);

HEAVY_TYPE(Syntax);
HEAVY_TYPE(OpGen);
HEAVY_TYPE(MlirValue);


#undef HEAVY_TYPE
}

#define GET_OP_CLASSES
#include "heavy/Ops.h.inc"

#endif
