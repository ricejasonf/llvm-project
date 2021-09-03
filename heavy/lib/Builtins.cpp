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
heavy::ExternFunction HEAVY_BASE_VAR(callcc);
heavy::ExternFunction HEAVY_BASE_VAR(with_exception_handler);
heavy::ExternFunction HEAVY_BASE_VAR(raise);
heavy::ExternFunction HEAVY_BASE_VAR(error);

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
  Value Current = P->Cdr;
  while (Value Node = Current.car()) {
    if (ImportSet* ImpSet = Context.CreateImportSet(Node)) {
      Context.Import(ImpSet);
    }
    Current = Current.cdr();
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
heavy::Value callcc(Context& C, ValueRefs Args) {
  unsigned Len = Args.size();
  assert(Len == 1 && "Invalid arity to builtin `callcc`");
  C.CallCC(Args[0]);
  return Undefined{};
}

heavy::Value eval(Context& C, ValueRefs Args) {
  unsigned Len = Args.size();
  assert((Len == 1 || Len == 2) && "Invalid arity to builtin `eval`");
  unsigned i = 0;
  Value EnvStack = (Len == 2) ? Args[i++] : nullptr;
  Value ExprOrDef = Args[i];

  // TODO We need to be able to pass a ModuleOp for insertion
  //      or create a temporary one if the Environment is immutable.
  //
  //      We could possible store the ModuleOp in the Environment object
  //
  if (Environment* E = dyn_cast_or_null<Environment>(EnvStack)) {
    // FIXME This modification of the EnvStack is incorrect I think
    //       It should probably just replace the EnvStack and revert
    //       when finished.
    llvm_unreachable("TODO");
    // nest the Environment in the EnvStack
    EnvStack = C.CreatePair(E);
  }

  mlir::Operation* Op = C.OpGen->VisitTopLevel(ExprOrDef);
  if (!Op) return Undefined{};

  if (C.CheckError()) return C.CreateUndefined();
  return opEval(C.OpEval, Op);
}

heavy::Value dump(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.SetError("invalid arity");
  Args[0].dump();
  C.Cont(heavy::Undefined());
  return Undefined();
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
  C.Cont(Temp);
  return Temp;
}

heavy::Value mul(Context&C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Mul>(C, Temp, X);
  }
  C.Cont(Temp);
  return Temp;
}

heavy::Value sub(Context&C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Sub>(C, Temp, X);
  }
  C.Cont(Temp);
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
  C.Cont(Temp);
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
  C.Cont(Bool(::heavy::eqv(V1, V2)));
  return Undefined();
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

heavy::Value with_exception_handler(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.SetError("invalid arity");
  C.WithExceptionHandler(Args[0], Args[1]);
  return Value();
}

heavy::Value raise(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.SetError("invalid arity");
  C.Raise(Args[0]);
  return Value();
}

heavy::Value error(Context& C, ValueRefs Args) {
  if (Args.size() == 0) return C.SetError("invalid arity");
  heavy::SourceLocation Loc;
  if (Args.size() > 1) {
    Loc = Args[1].getSourceLocation();
  }
  ValueRefs IrrArgs= Args.drop_front();
  Value IrrList = Empty();
  while (!IrrArgs.empty()) {
    IrrList = C.CreatePair(IrrArgs.front(), IrrList);
    IrrArgs = IrrArgs.drop_front();
  }
  Value Error = C.CreateError(Loc, Args[0], IrrList);
  C.Raise(Error);
  return Value();
}

}} // end of namespace heavy::base
