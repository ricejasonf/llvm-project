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

#include "heavy/Dialect.h"
#include "heavy/HeavyScheme.h"
#include "heavy/Source.h"
#include "heavy/OpGen.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using namespace heavy;

namespace {
heavy::SourceLocation getSourceLocation(mlir::Location Loc) {
  if (!Loc.isa<mlir::OpaqueLoc>()) return {};
  return heavy::SourceLocation(
    mlir::OpaqueLoc::getUnderlyingLocation<heavy::SourceLocationEncoding*>(
      Loc.cast<mlir::OpaqueLoc>()));
}

class OpEvaluator {
  heavy::Context& Context;
  llvm::DenseMap<mlir::Value, heavy::Value*> ValueMap;

  void setValue(mlir::Value M, heavy::Value* H) {
    ValueMap[M] = H;
  }

  heavy::Value* getValue(mlir::Value M) {
    heavy::Value* V = ValueMap.lookup(M);
    if (!V) {
      return Context.CreateUndefined();
    }
    return V;
  }

public:
  OpEvaluator(heavy::Context& C)
    : Context(C),
      ValueMap()
  { }

  heavy::Value* Visit(mlir::Value V) {
    return Visit(V.getDefiningOp());
  }

  heavy::Value* Visit(mlir::Operation* Op) {
    // FIXME there has to be a better way to do this
    if (llvm::isa<ApplyOp>(Op))   return Visit(llvm::cast<ApplyOp>(Op));
    if (llvm::isa<BindingOp>(Op)) return Visit(llvm::cast<BindingOp>(Op));
    if (llvm::isa<DefineOp>(Op))  return Visit(llvm::cast<DefineOp>(Op));
    if (llvm::isa<LiteralOp>(Op)) return Visit(llvm::cast<LiteralOp>(Op));
    llvm_unreachable("Unknown Operation");
  }

  heavy::Value* Visit(ApplyOp Op) {
    heavy::SourceLocation CallLoc = getSourceLocation(Op.getLoc());
    // The Callee should be a procedure or builtin
    unsigned ArgCount = Op.args().size() + 1; // includes callee
    heavy::StackFrame* Frame = Context.EvalStack.push(ArgCount, CallLoc);
    if (!Frame) return Context.CreateUndefined();

    llvm::MutableArrayRef<heavy::Value*> Args = Frame->getArgs();

    // We store args left to right, but we want to evaluate them
    // in reverse order to prevent accidental reliance on unspecified
    // behaviour
    for (int i = Op.args().size() - 1; i >= 0; --i) {
      Args[i] = Visit(Op.args()[i]);
    }

    heavy::Value* Callee = Visit(Op.fn());
    Frame->setCallee(Callee);

    heavy::Value* Result = Context.CreateUndefined();

    switch (Callee->getKind()) {
      case Value::Kind::Procedure:
        llvm_unreachable("TODO");
        break;
      case Value::Kind::Builtin: {
        Builtin* B = cast<Builtin>(Callee);
        Result = B->Fn(Context, Args);
      }
      default: {
        String* Msg = Context.CreateString(
          "invalid operator for call expression: ",
          Callee->getKindName()
        );
        Context.SetError(CallLoc, Msg, Callee);
      }
    }

    Context.EvalStack.pop();
    return Result;
  }

  heavy::Value* Visit(BindingOp Op) {
    return getValue(Op.result());
  }

  heavy::Value* Visit(LiteralOp Op) {
    // Map the IR value node to the run-time value
    heavy::Value* V = Op.input(); 
    setValue(Op.result(), V);
    return V;
  }

  heavy::Value* Visit(DefineOp Op) {
    // Evaluate the initializer expression
    // and assign it the the binding
    if (BindingOp B = dyn_cast<BindingOp>(Op.binding().getDefiningOp())) {
      assert(B && "DefineOp must contain BindingOp");
      setValue(Op.binding(), Visit(B.input()));
    }
    return Context.CreateUndefined();
  }
};
}

heavy::Value* opEval(heavy::Context& C, mlir::Value V) {
  OpEvaluator Eval(C);
  return Eval.Visit(V.getDefiningOp());
}
