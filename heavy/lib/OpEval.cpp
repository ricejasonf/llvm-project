//===---- OpGen.cpp - Classes for generating MLIR Operations ----*- C++ -*-===//
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

#include "heavy/HeavyScheme.h"
#include "heavy/EvaluationStack.h"
#include "heavy/OpGen.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using namespace heavy;

namespace {
class OpEvaluator {
  heavy::Context& Context;

  void push(Value* V) {
    Context.EvalStack.push(V);
  }
  Value* pop() {
    return Context.EvalStack.pop();
  }
  Value* top() {
    return Context.EvalStack.top();
  }

  llvm::DenseMap<mlir::Value*, heavy::Value> ValueMap = {};

  void setValue(mlir::Value* M, heavy::Value* H) {
    ValueMap[M] = H;
  }

  heavy::Value* getValue(mlir::Value* M) {
    heavy::Value* V = ValueMap.lookup(M);
    if (!V) {
      // not sure if this should ever happen
      return Context.CreateUndefined();
    }
    return V;
  }

public:
  OpEvaluator(heavy::Context& C)
    : Context(C)
  { }

  void Visit(mlir::Value V) {
    Visit(V.getDefiningOp());
  }

  void Visit(mlir::Operation* Op) {
    // FIXME there has to be a better way to do this
    if (llvm::isa<ApplyOp>(Op))   return Visit(llvm::cast<ApplyOp>(Op));
    if (llvm::isa<BindingOp>(Op)) return Visit(llvm::cast<BindingOp>(Op));
    if (llvm::isa<DefineOp>(Op))  return Visit(llvm::cast<DefineOp>(Op));
    if (llvm::isa<LiteralOp>(Op)) return Visit(llvm::cast<LiteralOp>(Op));
    llvm_unreachable("Unknown Operation");
  }

  void Visit(ApplyOp Op) {
    // TODO make a proper call stack
    Visit(Op.fn())
    for (mlir::Operation* X : Op.args()) {
      Visit(X);
    }
  }

  void Visit(BindingOp Op) {
    push(getValue(Op.result()));
  }

  void Visit(LiteralOp Op) {
    if (Context.CheckError()) return;
    // Map the IR value node to the run-time value
    heavy::Value* V = Op.input(); 
    setValue(Op.result(), V);
    push(V);
  }

  void Visit(DefineOp Op) {
    if (Context.CheckError()) return;
    // Evaluate the initializer expression
    Visit(Op.input().getDefiningOp());
    push(Context.CreateUndefined());
  }
};
}

void opEval(heavy::Context& C, mlir::Value V) {
  OpEvaluator Eval(C);
  Eval.Visit(V.getDefiningOp());
}
