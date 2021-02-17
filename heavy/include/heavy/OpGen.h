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
#include <utility>

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
  mlir::OpBuilder TopLevel;
  mlir::OpBuilder LocalInits;
  mlir::OpBuilder LocalSequence;
  llvm::ScopedHashTable<heavy::Binding*, mlir::Value> BindingTable;
  llvm::SmallVector<heavy::Binding*> LocalDefines;
  bool IsTopLevel = false;

  using BindingScope = llvm::ScopedHashTableScope<
                                            heavy::Binding*,
                                            mlir::Value>;

public:
  explicit OpGen(heavy::Context& C);

  mlir::Value VisitTopLevel(Value* V) {
    // there should be an insertion point already setup
    // for the module init function
    IsTopLevel = true;
    return Visit(V);
  }

  bool isTopLevel() { return IsTopLevel; }
  bool isLocalDefineAllowed();

  template <typename Op, typename ...Args>
  static Op create(OpBuilder& Builder, heavy::SourceLocation Loc,
                   Args&& ...args) {
    assert(Builder.getInsertionBlock() != nullptr &&
        "Operation must have insertion point");
    mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                               Builder.getContext());
    return Builder.create<Op>(MLoc, std::forward<Args>(args)...);
  }

  template <typename Op, typename ...Args>
  Op create(heavy::SourceLocation Loc, Args&& ...args) {
    return create<Op>(Builder, Loc, std::forward<Args>(args));
  }

  mlir::Value createLambda(Value* Formals, Value* Body,
                           SourceLocation Loc,
                           llvm::StringRef Name = {});
          
  mlir::Value createDefine(Symbol* S, Value *V, Value* OrigCall);
  mlir::Value createTopLevelDefine(Symbol* S, mlir::Value Init,
                                   Value* OrigCall);
  mlir::Value createTopLevelDefine(Symbol* S, mlir::Value Init, Module* M);

  template <typename T>
  mlir::Value SetError(T Str, Value* V) {
    Context.SetError(Str, V);
    return Error(V->getSourceLocation());
  }

  mlir::Value Error(SourceLocation Loc) {
    return create<LiteralOp>(Loc, Context.CreateUndefined());
  }

private:
  mlir::Value VisitValue(Value* V) {
    return create<LiteralOp>(V->getSourceLocation(), V);
  }

  mlir::Value VisitBuiltin(Builtin* V) {
    return create<BuiltinOp>(V->getSourceLocation(), V);
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
