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
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
  class Value;
}

namespace heavy {

heavy::Value* opEval(Context&, mlir::Value);

class OpEval;

class OpGen : public ValueVisitor<OpGen, mlir::Value> {
  friend class ValueVisitor<OpGen, mlir::Value>;
  heavy::Context& Context;
  llvm::DenseMap<heavy::Binding*, mlir::Value> BindingTable;
  mlir::OpBuilder Builder;

public:
  bool IsTopLevel = false;

  explicit OpGen(heavy::Context& C);

  mlir::Value VisitTopLevel(Value* V) {
    IsTopLevel = true;
    return Visit(V);
  }

  template <typename Op, typename ...Args>
  mlir::Value create(heavy::SourceLocation Loc, Args&& ...args) {
    mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                               Builder.getContext());
    return Builder.create<Op>(MLoc,
                              std::forward<Args>(args)...);
  }

  mlir::Value createTopLevelDefine(Symbol* S, Value *V, Value* OrigCall);

  template <typename T>
  mlir::Value SetError(T Str, Value* V) {
    Context.SetError(Str, V);
    return create<LiteralOp>(V->getSourceLocation(),
                             Context.CreateUndefined());
  }

private:
  mlir::Value VisitValue(Value* V) {
    return create<LiteralOp>(V->getSourceLocation(), V);
  }

  mlir::Value VisitSymbol(Symbol* S);

  mlir::Value HandleCall(Pair* P);
  void HandleCallArgs(Value *V,
                      llvm::SmallVectorImpl<mlir::Value>& Args);

  mlir::Value VisitPair(Pair* P);
  // TODO mlir::Value VisitVector(Vector* V);
};

}

#endif
