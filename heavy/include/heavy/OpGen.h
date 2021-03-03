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
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"
#include <utility>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
  class Value;
}

namespace heavy {

class OpGen : public ValueVisitor<OpGen, mlir::Value> {
  friend class ValueVisitor<OpGen, mlir::Value>;
  using BindingScopeTable = llvm::ScopedHashTable<
                                            heavy::Binding*,
                                            mlir::Value>;
  using BindingScope = typename BindingScopeTable::ScopeTy;

  heavy::Context& Context;
  mlir::OpBuilder Builder;
  mlir::OpBuilder LocalInits;
  BindingScopeTable BindingTable;
  BindingScope BindingTableTop;
  mlir::Operation* TopLevel;
  bool IsTopLevel = false;
  bool IsTailPos = true;

  struct TailPosScope {
    bool& State;
    bool PrevState;

    TailPosScope(heavy::OpGen& OpGen)
      : State(OpGen.IsTailPos),
        PrevState(State)
    { }
      
    TailPosScope(TailPosScope&) = delete;

    ~TailPosScope() {
      State = PrevState;
    }
  };

public:
  explicit OpGen(heavy::Context& C);

  mlir::ModuleOp getTopLevel();

  mlir::Value VisitTopLevel(Value* V) {
    // there should be an insertion point already setup
    // for the module init function
    IsTopLevel = true;
    return Visit(V);
  }

  bool isTopLevel() { return IsTopLevel; }
  bool isTailPos() { return IsTailPos && !IsTopLevel; }
  bool isLocalDefineAllowed();

  template <typename Op, typename ...Args>
  static Op create(mlir::OpBuilder& Builder, heavy::SourceLocation Loc,
                   Args&& ...args) {
    assert(Builder.getInsertionBlock() != nullptr &&
        "Operation must have insertion point");
    mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                               Builder.getContext());
    return Builder.create<Op>(MLoc, std::forward<Args>(args)...);
  }

  template <typename Op, typename ...Args>
  Op create(heavy::SourceLocation Loc, Args&& ...args) {
    return create<Op>(Builder, Loc, std::forward<Args>(args)...);
  }

  void processBody(SourceLocation Loc, Value* Body);
  void processSequence(SourceLocation Loc, Value* Body);

  mlir::Value createLambda(Value* Formals, Value* Body,
                           SourceLocation Loc,
                           llvm::StringRef Name = {});

  mlir::Value createBinding(Binding *B, mlir::Value Init);
  mlir::Value createDefine(Symbol* S, Value *Args, Value* OrigCall);
  mlir::Value createTopLevelDefine(Symbol* S, Value* Args, Value* OrigCall);
  mlir::Value createTopLevelDefine(Symbol* S, mlir::Value Init, Module* M);
  mlir::Value createUndefined();

  template <typename T>
  mlir::Value SetError(T Str, Value* V) {
    Context.SetError(Str, V);
    return Error();
  }

  template <typename T>
  mlir::Value SetError(SourceLocation Loc, T Str, Value* V) {
    Context.SetError(Loc, Str, V);
    return Error();
  }

  mlir::Value Error() {
    return createUndefined();
  }

private:
  mlir::Value VisitDefineArgs(Value* Args);

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
