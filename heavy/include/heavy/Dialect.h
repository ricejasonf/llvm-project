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

struct Dialect : public mlir::Dialect {
  explicit Dialect(mlir::MLIRContext* Ctx);
  static llvm::StringRef getDialectNamespace() { return "heavy"; }

#if 0
  mlir::Attribute parseAttribute(mlir::DialectAsmParser& P,
                                 mlir::Type type) const override;
#endif
  mlir::Type parseType(mlir::DialectAsmParser& P) const override;

  mlir::Attribute parseAttribute(mlir::DialectAsmParser& P,
                                 mlir::Type Type) const override;
  void printAttribute(mlir::Attribute Attr,
                      mlir::DialectAsmPrinter& P) const override;
  void printType(mlir::Type, mlir::DialectAsmPrinter&) const override;
};

struct HeavyValueTy : public mlir::Type::TypeBase<
                            HeavyValueTy,
                            mlir::Type,
                            mlir::TypeStorage> {
  static constexpr llvm::StringLiteral name = "heavy.value";
  static constexpr llvm::StringRef getMnemonic() { return "value"; }
  using Base::Base;
};

struct HeavyRestTy : public mlir::Type::TypeBase<
                            HeavyRestTy,
                            mlir::Type,
                            mlir::TypeStorage> {
  static constexpr llvm::StringLiteral name = "heavy.rest";
  static constexpr llvm::StringRef getMnemonic() { return "rest"; }
  using Base::Base;
};

class HeavyValueAttrStorage;

// Declare HeavyValueAttr manually.
class HeavyValueAttr : public mlir::Attribute::AttrBase<
                            HeavyValueAttr,
                            mlir::Attribute,
                            HeavyValueAttrStorage> {
  using Base::Base;

public:
  static constexpr llvm::StringLiteral name = "heavy.value_attr";

  static HeavyValueAttr get(mlir::MLIRContext*, StringAttr expr);
  static HeavyValueAttr get(mlir::MLIRContext*, heavy::Value Val);

  // getCachedValue - For garbage collector access.
  heavy::Value& getCachedValue();
  heavy::Value getValue(heavy::Context& C) const;
  heavy::StringAttr getExpr() const;
};
} //  namespace heavy
MLIR_DECLARE_EXPLICIT_TYPE_ID(heavy::HeavyValueAttr)

namespace heavy {
// Additional Types
#define HEAVY_TYPE(NAME, PRINT_NAME, MNEMONIC) \
struct Heavy##NAME##Ty : public mlir::Type::TypeBase< \
Heavy##NAME##Ty, mlir::Type, mlir::TypeStorage> { \
  static constexpr llvm::StringLiteral name = PRINT_NAME; \
  static constexpr llvm::StringRef getMnemonic() { return MNEMONIC; } \
  using Base::Base; \
} \

HEAVY_TYPE(Pair, "heavy.pair", "pair");

HEAVY_TYPE(Syntax, "heavy.syntax", "syntax");
HEAVY_TYPE(OpGen, "heavy.opgen", "opgen");
HEAVY_TYPE(MlirValue, "heavy.mlir_value", "mlir_value");


#undef HEAVY_TYPE
}

#define GET_OP_CLASSES
#include "heavy/Ops.h.inc"

#endif
