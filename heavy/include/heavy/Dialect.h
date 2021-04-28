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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

// Kinds go away with an August 2020 MLIR commit in `main`
#define HEAVY_VALUE_KIND \
  mlir::Attribute::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_ATTR

// In Value we assume Operation* has the same alignment as ValueBase*.
// (which should be 8 bytes)
// Check that here
static_assert(llvm::PointerLikeTypeTraits<heavy::ValueBase*>::NumLowBitsAvailable ==
              llvm::PointerLikeTypeTraits<heavy::Operation*>::NumLowBitsAvailable,
              "mlir::Operation* must have 8 byte alignment to fit in heavy::Value");

namespace heavy {
using mlir::FuncOp;
}

namespace mlir {
namespace heavy_mlir {

struct Dialect : public mlir::Dialect {
  explicit Dialect(mlir::MLIRContext* Ctx);
  static llvm::StringRef getDialectNamespace() { return "heavy"; }

  void printAttribute(mlir::Attribute Attr,
                      mlir::DialectAsmPrinter& P) const override;
  void printType(mlir::Type, mlir::DialectAsmPrinter&) const override;
};

struct DialectRegisterer {
  DialectRegisterer() {
    mlir::registerDialect<Dialect>();
  }
};

struct HeavyValue : public mlir::Type::TypeBase<
                            HeavyValue,
                            mlir::Type,
                            mlir::TypeStorage> {
  using Base::Base;


  // This will go away
  static HeavyValue get(mlir::MLIRContext* C) {
    return Base::get(C, HEAVY_VALUE_KIND);
  }
};

struct HeavyRest : public mlir::Type::TypeBase<
                            HeavyRest,
                            mlir::Type,
                            mlir::TypeStorage> {
  using Base::Base;


  // This will go away
  static HeavyRest get(mlir::MLIRContext* C) {
    return Base::get(C, HEAVY_VALUE_KIND);
  }
};

struct HeavyLambda : public mlir::Type::TypeBase<
                            HeavyLambda,
                            mlir::Type,
                            mlir::TypeStorage> {
  using Base::Base;


  // This will go away
  static HeavyLambda get(mlir::MLIRContext* C) {
    return Base::get(C, HEAVY_VALUE_KIND);
  }
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

  static bool kindof(unsigned K) {
    return K == HEAVY_VALUE_KIND;
  }

  static HeavyValueAttr get(MLIRContext* C, heavy::Value V) {
    return Base::get(C, HEAVY_VALUE_KIND, V);
  }
};

#define GET_OP_CLASSES
#include "heavy/Ops.h.inc"
}
}

namespace heavy {
  using namespace mlir::heavy_mlir;
}

#endif
