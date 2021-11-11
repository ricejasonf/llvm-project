//===-- TemplateBase.h - Class for Syntax Pattern Matching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Define heavy::TemplateBase mapping heavy::Value to mlir::Value
//  for user defined syntax transformations via the `syntax-rules` syntax.
//  This file is provided as header-only to support OpGen.cpp.
//
//===----------------------------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SmallVector.h"

namespace heavy {

// TemplateBase
//    - Provide a base class for quasiquote and syntax-rules
//      templates. This holds the reference to OpGen and provides
//      utility functions for creating operations that build
//      literals interspersed with values.
template <typename Derived>
class TemplateBase : ValueVisitor<Derived, heavy::Value> {
protected:
  using Base = ValueVisitor<Derived, heavy::Value>;

  heavy::OpGen& OpGen;

  TemplateBase(OpGen& O)
    : OpGen(O)
  { }

  using Base::getDerived;

  bool isRebuilt(heavy::Value V) {
    return (V.is<ValueSumType::Operation>() ||
            V.is<ValueSumType::ContArg>());
  }

  bool isRebuilt(heavy::Value V1, heavy::Value V2) {
    return isRebuilt(V1) || isRebuilt(V2);
  }

  // createValue - Create a mlir::Value from its stored
  //               representation in heavy::Value or wrap
  //               it in a LiteralOp.
  mlir::Value createValue(heavy::Value V) {
    mlir::Value Val = OpGen::toValue(V);
    if (Val) return OpGen.LocalizeValue(Val);
    return createLiteral(V);
  }

  heavy::Value setError(llvm::StringRef S, heavy::Value V) {
    OpGen.getContext().SetError(S, V);
    return Undefined();
  }

  mlir::Operation* createLiteral(Value V) {
    return OpGen.create<LiteralOp>(V.getSourceLocation(), V);
  }

  mlir::Operation* createList(SourceLocation Loc, heavy::Value X,
                              heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    mlir::Value Empty = createLiteral(heavy::Empty{});
    return OpGen.create<ConsOp>(Loc, ValX,
        OpGen.create<ConsOp>(Loc, ValY, Empty));
  }

  mlir::Operation* createCons(SourceLocation Loc, heavy::Value X,
                                                  heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    return OpGen.create<ConsOp>(Loc, ValX, ValY);
  }

  mlir::Operation* createSplice(SourceLocation Loc, heavy::Value X,
                                heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    return OpGen.create<SpliceOp>(Loc, ValX, ValY);
  }
};

}
