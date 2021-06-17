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

#include "heavy/Builtins.h"
#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "llvm/Support/Casting.h"

bool HEAVY_BASE_IS_LOADED = false;

// import must be pre-loaded
heavy::ExternSyntax HEAVY_BASE_VAR(import);

heavy::ExternSyntax HEAVY_BASE_VAR(define);
heavy::ExternSyntax HEAVY_BASE_VAR(if);
heavy::ExternSyntax HEAVY_BASE_VAR(lambda);
heavy::ExternSyntax HEAVY_BASE_VAR(quasiquote);
heavy::ExternSyntax HEAVY_BASE_VAR(quote);
heavy::ExternSyntax HEAVY_BASE_VAR(set);

heavy::ExternFunction HEAVY_BASE_VAR(add);
heavy::ExternFunction HEAVY_BASE_VAR(sub);
heavy::ExternFunction HEAVY_BASE_VAR(div);
heavy::ExternFunction HEAVY_BASE_VAR(mul);
heavy::ExternFunction HEAVY_BASE_VAR(gt);
heavy::ExternFunction HEAVY_BASE_VAR(lt);
heavy::ExternFunction HEAVY_BASE_VAR(list);
heavy::ExternFunction HEAVY_BASE_VAR(append);
heavy::ExternFunction HEAVY_BASE_VAR(dump);
heavy::ExternFunction HEAVY_BASE_VAR(eq);
heavy::ExternFunction HEAVY_BASE_VAR(equal);
heavy::ExternFunction HEAVY_BASE_VAR(eqv);
heavy::ExternFunction HEAVY_BASE_VAR(eval);

namespace heavy { namespace base {

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
  Value Formals = P2->Car;
  Pair* Body = dyn_cast<Pair>(P2->Cdr);

  return OG.createLambda(Formals, Body, P->getSourceLocation());
}

mlir::Value if_(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value CondExpr = P2->Car;
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value ThenExpr = P2->Car;
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value ElseExpr = P2->Car;
  if (!isa<Empty>(P2->Cdr)) {
    return OG.SetError("invalid if syntax", P);
  }
  return OG.createIf(P->getSourceLocation(), CondExpr,
                     ThenExpr, ElseExpr);
}

mlir::Value set(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid set syntax", P);
  heavy::Symbol* S = dyn_cast<Symbol>(P2->Car);
  if (!S) return OG.SetError("expecting symbol", P2);
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid set syntax", P2);
  heavy::Value Expr = P2->Car;
  if (!isa<Empty>(P2->Cdr)) return OG.SetError("invalid set syntax", P2);
  return OG.createSet(P->getSourceLocation(), S, Expr);
}

mlir::Value import(OpGen& OG, Pair* P) {
  heavy::Context& Context = OG.getContext(); 
  ImportSet* ImpSet = Context.CreateImportSet(P->Cdr.car());
  if (ImpSet) {
    Context.Import(ImpSet);
  }
  return OG.createUndefined();
}

}} // end of namespace heavy::base

namespace heavy {
// TODO Replace NumberOp here with corresponding arithmetic ops in OpGen and OpEval
struct NumberOp {
  // These operations always mutate the first operand
  struct Add {
    static Int f(Int X, Int Y) { return X + Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val + Y->Val; }
  };
  struct Sub {
    static Int f(Int X, Int Y) { return X - Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val - Y->Val; }
  };
  struct Mul {
    static Int f(Int X, Int Y) { return X * Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val * Y->Val; }
  };
  struct Div {
    static Int f(Int X, Int Y) { return X / Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val / Y->Val; }
  };
};

} // end namespace heavy

namespace heavy { namespace base {
heavy::Value eval(Context& C, ValueRefs Args) {
  unsigned Len = Args.size();
  assert((Len == 1 || Len == 2) && "Invalid arity to builtin `eval`");
  unsigned i = 0;
  Value EnvStack = (Len == 2) ? Args[i++] : nullptr;
  Value ExprOrDef = Args[i];
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

heavy::Value dump(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.SetError("invalid arity");
  Args[0].dump();
  return heavy::Undefined();
}

template <typename Op>
heavy::Value operator_helper(Context& C, Value X, Value Y) {
  if (!X.isNumber()) return C.SetInvalidKind(X);
  if (!Y.isNumber()) return C.SetInvalidKind(Y);
  ValueKind CommonKind = Number::CommonKind(X, Y);
  switch (CommonKind) {
    case ValueKind::Float: {
      llvm_unreachable("TODO casting to float");
      return nullptr;
    }
    case ValueKind::Int: {
      // we can assume they are both Int
      return Op::f(cast<Int>(X), cast<Int>(Y));
    }
    default:
      llvm_unreachable("invalid arithmetic type");
  }
}

heavy::Value add(Context& C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Add>(C, Temp, X);
  }
  return Temp;
}

heavy::Value mul(Context&C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Mul>(C, Temp, X);
  }
  return Temp;
}

heavy::Value sub(Context&C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Sub>(C, Temp, X);
  }
  return Temp;
}

heavy::Value div(Context& C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    if (Number::isExactZero(X)) {
      C.SetError("divide by exact zero", X);
      return X;
    }
    Temp = operator_helper<NumberOp::Div>(C, Temp, X);
  }
  return Temp;
}

heavy::Value gt(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO");
  return nullptr;
}

heavy::Value lt(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO");
  return nullptr;
}

heavy::Value equal(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.SetError("invalid arity");
  Value V1 = Args[0];
  Value V2 = Args[1];
  return Bool(::heavy::equal(V1, V2));
}

heavy::Value eqv(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.SetError("invalid arity");
  Value V1 = Args[0];
  Value V2 = Args[1];
  return Bool(::heavy::eqv(V1, V2));
}

heavy::Value list(Context& C, ValueRefs Args) {
  // Returns a *newly allocated* list of its arguments.
  heavy::Value List = C.CreateEmpty();
  for (heavy::Value Arg : Args) {
    List = C.CreatePair(Arg, List);
  }
  return List;
}

heavy::Value append(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO append");
}

}} // end of namespace heavy::base
