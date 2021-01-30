//===- Builtins.h - Builtin functions for HeavyScheme -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines HeavyScheme decalarations for values and evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_BUILTINS_H
#define LLVM_HEAVY_BUILTINS_H

#include "heavy/HeavyScheme.h"
#include "heavy/OpEval.h"
#include "heavy/OpGen.h"
#include "llvm/Support/Casting.h"

namespace heavy {
// GetSingleArg - Given a macro expression (keyword datum)
//                return the first argument iff there is only
//                one argument otherwise returns nullptr
//                (same as `cadr`)
Value* GetSingleSyntaxArg(Pair* P) {
  // P->Car is the syntactic keyword
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (P2 && isa<Empty>(P2->Cdr)) {
    return P2->Car;
  }
  return nullptr;
}

}

namespace heavy { namespace builtin_syntax {

mlir::Value define(OpGen& OG, Pair* P) {
  Pair*   P2  = dyn_cast<Pair>(P->Cdr);
  Symbol* S   = nullptr;
  Value*  V   = nullptr;
  if (!P2) return OG.SetError("invalid define syntax", P);
  if (Pair* LambdaSpec = dyn_cast<Pair>(P2->Car)) {
    llvm_unreachable("TODO");
    S = dyn_cast<Symbol>(LambdaSpec->Car);
#if 0 // CheckLambda would return nullptr
    V = C.CheckLambda(/*LambdaParams=*/LambdaSpec->Cdr,
                      /*LambdaBody=*/P2->Cdr,
                      /*LambdaName=*/S);
#endif
  } else {
    S = dyn_cast<Symbol>(P2->Car);
    V = GetSingleSyntaxArg(P2);
  }
  if (!S || !V) return OG.SetError("invalid define syntax", P);
  if (OG.IsTopLevel) {
    return OG.createTopLevelDefine(S, V, P);
  } else {
    // Handle internal definitions inside
    // lambda syntax
    return OG.SetError("unexpected define", P);
  }
}

mlir::Value quote(OpGen& OG, Pair* P) {
  Value* Arg = GetSingleSyntaxArg(P);
  if (!Arg) {
    return OG.SetError("invalid quote syntax", P);
  }

  return OG.create<LiteralOp>(P->getSourceLocation(), Arg);
}

mlir::Value quasiquote(OpGen& C, Pair* P) {
  llvm_unreachable("TODO migrate Quasiquoter to OpGen");
#if 0
  Quasiquoter QQ(C);
  return QQ.Run(P);
#endif
  return {}; // ??
}

}} // end of namespace heavy::builtin_syntax

namespace heavy {
// TODO Replace NumberOp here with corresponding arithmetic ops in OpGen and OpEval
struct NumberOp {
  // These operations always mutate the first operand
  struct Add {
    static void f(Integer* X, Integer* Y) { X->Val += Y->Val; }
    static void f(Float* X, Float *Y) { X->Val = X->Val + Y->Val; }
  };
  struct Sub {
    static void f(Integer* X, Integer* Y) { X->Val -= Y->Val; }
    static void f(Float* X, Float *Y) { X->Val = X->Val - Y->Val; }
  };
  struct Mul {
    static void f(Integer* X, Integer* Y) { X->Val *= Y->Val; }
    static void f(Float* X, Float *Y) { X->Val = X->Val * Y->Val; }
  };
  struct Div {
    static void f(Integer* X, Integer* Y) { X-> Val = X->Val.sdiv(Y->Val); }
    static void f(Float* X, Float *Y) { X->Val = X->Val / Y->Val; }
  };
};

} // end namespace heavy

namespace heavy { namespace builtin {
heavy::Value* eval(Context& C, ValueRefs Args) {
  unsigned Len = Args.size();
  assert((Len == 1 || Len == 2) && "Invalid arity to builtin `eval`");
  unsigned i = 0;
  Value* EnvStack = (Len == 2) ? Args[i++] : nullptr;
  Value* ExprOrDef = Args[i];
  if (Environment* E = dyn_cast_or_null<Environment>(EnvStack)) {
    // nest the Environment in the EnvStack
    EnvStack = C.CreatePair(E);
  }

  mlir::Value OpGenResult = C.OpGen->VisitTopLevel(ExprOrDef);
  if (C.CheckError()) return C.CreateUndefined();
  OpGenResult.dump();
  return C.OpEval->Visit(OpGenResult);
}

template <typename Op>
heavy::Value* operator_helper(Context& C, Value* XVal, Value *YVal) {
  Number* X = C.CheckKind<Number>(XVal);
  if (C.CheckError()) return C.CreateUndefined();
  Number* Y = C.CheckKind<Number>(YVal);
  if (C.CheckError()) return C.CreateUndefined();
  Value::Kind CommonKind = Number::CommonKind(X, Y);
  Number* Result;
  switch (CommonKind) {
    case Value::Kind::Float: {
      Float* CopyX = C.CreateFloat(cast<Float>(X)->getVal());
      Float* CopyY = C.CreateFloat(cast<Float>(Y)->getVal());
      Op::f(CopyX, CopyY);
      Result = CopyX;
      break;
    }
    case Value::Kind::Integer: {
      // we can assume they are both integers
      Integer* CopyX = C.CreateInteger(cast<Integer>(X)->getVal());
      Op::f(CopyX, cast<Integer>(Y));
      Result = CopyX;
      break;
    }
    default:
      llvm_unreachable("Invalid arithmetic type");
  }
  return Result;
}

heavy::Value* operator_add(Context& C, ValueRefs Args) {
  Value* Temp = Args[0];
  for (heavy::Value* X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Add>(C, Temp, X);
  }
  return Temp;
}

heavy::Value* operator_mul(Context&C, ValueRefs Args) {
  Value* Temp = Args[0];
  for (heavy::Value* X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Mul>(C, Temp, X);
  }
  return Temp;
}

heavy::Value* operator_sub(Context&C, ValueRefs Args) {
  Value* Temp = Args[0];
  for (heavy::Value* X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Sub>(C, Temp, X);
  }
  return Temp;
}

heavy::Value* operator_div(Context& C, ValueRefs Args) {
  Value* Temp = Args[0];
  for (heavy::Value* X : Args.drop_front()) {
    if (Number::isExactZero(X)) {
      C.SetError("divide by exact zero", X);
      return X;
    }
    Temp = operator_helper<NumberOp::Div>(C, Temp, X);
  }
  return Temp;
}

heavy::Value* list(Context& C, ValueRefs Args) {
  // Returns a *newly allocated* list of its arguments.
  heavy::Value* List = C.CreateEmpty();
  for (heavy::Value* Arg : Args) {
    List = C.CreatePair(Arg, List);
  }
  return List;
}

heavy::Value* append(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO append");
}

}} // end of namespace heavy::builtin

#endif

