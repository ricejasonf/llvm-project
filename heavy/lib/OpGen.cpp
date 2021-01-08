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
#include "heavy/OpGen.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using namespace heavy;

mlir::OwningModuleRef opGen(Context&, Value* V, Value* EnvStack) {
  OpGen O(C, EnvStack);
  return O.Visit(V);
}

class OpGen : public ValueVisitor<OpGen, mlir::Value*> {
  friend class ValueVisitor<OpGen, Value*>;
  heavy::Context& Context;
  // heavy::Context::Lookup returns a heavy::Binding that
  // we want a corresponding Operation or mlir::Value
  llvm::DenseMap<heavy::Binding*, mlir::Value> BindingTable;

  mlir::ModuleOp Module;
  mlir::OpBuilder Builder;

public:
  OpGen(heavy::Context& C)
    : Context(C),
      MlirContext(C.MlirContext),
      IsTopLevel(false)
  { }

  Value* VisitTopLevel(Value* V) {
    IsTopLevel = true;
    return Visit(V);
  }

private:
  Value* VisitValue(Value* V) {
    // TODO constant values should be wrapped
    //      in a ConstantOp
    //      These should only be literals here
    return V;
  }

  mlir::Value VisitSymbol(Symbol* S) {
    // TODO return variable reference
    if (Binding* B = Context.Lookup(S)) return B;

    String* Msg = Context.CreateString("unbound symbol: ", S->getVal());
    Context.SetError(Msg, S);
    return Context.CreateUndefined();
  }

  mlir::Value HandleCallArgs(Value *V) {
    if (isa<Empty>(V)) return V;
    if (!isa<Pair>(V)) {
      return Context.SetError("improper list as call expression", V);
    }
    Pair* P = cast<Pair>(V);
    Value* CarResult = Visit(P->Car);
    Value* CdrResult = HandleCallArgs(P->Cdr);
    // TODO
    // Create a CallExpr AST node with a
    // known number of arguments instead of
    // recreating lists
    return Context.CreatePair(CarResult, CdrResult);
  }

  mlir::Value VisitPair(Pair* P) {
    if (Context.CheckError()) return Context.CreateEmpty();
    Binding* B = Context.Lookup(P->Car);
    if (!B) return P;

    // Operator might be some kind of syntax transformer
    Value* Operator = B->getValue();

    switch (Operator->getKind()) {
      case Value::Kind::BuiltinSyntax: {
        BuiltinSyntax* BS = cast<BuiltinSyntax>(Operator);
        return BS->Fn(Context, P);
      }
      case Value::Kind::Syntax:
        llvm_unreachable("TODO");
        return Context.CreateEmpty();
      default:
        IsTopLevel = false;
        return HandleCallArgs(P);
    }
  }

  mlir::Value VisitVector(Vector* V) {
    llvm::ArrayRef<Value*> Xs = V->getElements();
    Vector* New = Context.CreateVector(Xs.size());
    llvm::MutableArrayRef<Value*> Ys = New->getElements();
    for (unsigned i = 0; i < Xs.size(); ++i) {
      Visit(Xs[i]);
      Ys[i] = Visit(Xs[i]);
    } 
    return New;
  }
};

namespace heavy { namespace builtin_syntax {

mlir::Value define(Context& C, Pair* P) {
  Pair*   P2  = dyn_cast<Pair>(P->Cdr);
  Symbol* S   = nullptr;
  Value*  V   = nullptr;
  if (!P2) return C.SetError("invalid define syntax", P);
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
  if (!S || !V) return C.SetError("invalid define syntax", P);
  if (C.IsTopLevel) {
    // TODO build a DefineOp and add it to the map of bindings to ops
    return C.CreatePair(C.GetBuiltin("__define"),
            C.CreatePair(C.CreateQuote(
              C.CreateGlobal(S, V, P))));
  } else {
    // Handle internal definitions inside
    // lambda syntax
    return C.SetError("unexpected define", P);
  }
}

mlir::Value quote(Context& C, Pair* P) {
  Value* Result = GetSingleSyntaxArg(P);
  if (!Result) {
    C.SetError("invalid quote syntax", P);
    return C.CreateEmpty();
  }

  return C.CreateQuote(Result);
}

mlir::Value quasiquote(Context& C, Pair* P) {
  llvm_unreachable("TODO migrate Quasiquoter to OpGen");
#if 0
  Quasiquoter QQ(C);
  return QQ.Run(P);
#endif
  return {}; // ??
}

}} // end of namespace heavy::builtin_syntax

}

#endif
