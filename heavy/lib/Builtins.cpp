//===- Builtins.cpp - Builtin functions for HeavyScheme ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines builtins and builtin syntax for HeavyScheme
//
//===----------------------------------------------------------------------===//

#include "heavy/HeavyScheme.h"
#include "heavy/OpGen.h"
#include "llvm/Support/Casting.h"

namespace {
// GetSingleSyntaxArg
//              - Given a macro expression (keyword datum)
//                return the first argument iff there is only
//                one argument otherwise returns nullptr
//                (same as `cadr`)
heavy::Value* GetSingleSyntaxArg(heavy::Pair* P) {
  // P->Car is the syntactic keyword
  heavy::Pair* P2 = llvm::dyn_cast<heavy::Pair>(P->Cdr);
  if (P2 && llvm::isa<heavy::Empty>(P2->Cdr)) {
    return P2->Car;
  }
  return nullptr;
}

}

namespace heavy { namespace builtin_syntax {

mlir::Value define(OpGen& OG, Pair* P) {
  Pair*   P2    = dyn_cast<Pair>(P->Cdr);
  Symbol* S     = nullptr;
  if (!P2) return OG.SetError("invalid define syntax", P);
  if (Pair* LambdaSpec = dyn_cast<Pair>(P2->Car)) {
    S = dyn_cast<Symbol>(LambdaSpec->Car);

  } else {
    S = dyn_cast<Symbol>(P2->Car);
  }
  if (!S) return OG.SetError("invalid define syntax", P);
  return OG.createDefine(S, P2, P);
}

mlir::Value lambda(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  Value* Formals = P2->Car;
  Pair* Body = dyn_cast<Pair>(P2->Cdr);

  return OG.createLambda(Formals, Body, P->getSourceLocation());
}

mlir::Value if_(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value* CondExpr = P2->Car;
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value* ThenExpr = P2->Car;
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value* ElseExpr = P2->Car;
  if (!isa<Empty>(P2->Cdr)) {
    return OG.SetError("invalid if syntax", P);
  }
  return OG.createIf(P->getSourceLocation(), CondExpr,
                     ThenExpr, ElseExpr);
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

  // TODO
  // uhh we didn't change anything about where we are
  // inserting operations in relation to the "environment"
  // We need to set the insertion point to the relevant ModuleOp
  // inside "VisitTopLevel"
  // (I think heavy::Module needs to be upgraded to wrap ModuleOp
  //  or something)
  C.OpGen->VisitTopLevel(ExprOrDef);
  if (C.CheckError()) return C.CreateUndefined();
  return opEval(C.OpEval);
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

heavy::Value* operator_gt(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO");
  return nullptr;
}

heavy::Value* operator_lt(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO");
  return nullptr;
}

heavy::Value* eqv(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.SetError("invalid arity");
  Value* V1 = Args[0];
  Value* V2 = Args[1];
  if (V1 == V2) return C.CreateBoolean(true);
  if (V1->getKind() != V2->getKind()) {
    return C.CreateBoolean(false);
  }

  bool R;
  switch (V1->getKind()) {
  case Value::Kind::Symbol:
      R = cast<Symbol>(V1)->equals(
          cast<Symbol>(V2));
      break;
    // TODO For primitives this is temporary until
    // they are embedded in the pointers
  case Value::Kind::Boolean:
      R = cast<Boolean>(V1)->getVal() ==
          cast<Boolean>(V2)->getVal();
      break;
  case Value::Kind::Char:
      R = cast<Char>(V1)->getVal() ==
          cast<Char>(V2)->getVal();
      break;
  case Value::Kind::Integer:
      R = cast<Integer>(V1)->getVal() ==
          cast<Integer>(V2)->getVal();
      break;
  case Value::Kind::Float:
      R = cast<Float>(V1)->getVal() ==
          cast<Float>(V2)->getVal();
      break;
  case Value::Kind::Empty:
      R = true;
      break;
  default:
      R = false;
  }
  return C.CreateBoolean(R);
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

