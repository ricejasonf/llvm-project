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
#include "heavy/OpGen.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using namespace heavy;


class heavy::OpGen : public ValueVisitor<OpGen, mlir::Value> {
  friend class ValueVisitor<OpGen, mlir::Value>;
  heavy::Context& Context;
  // heavy::Context::Lookup returns a heavy::Binding that
  // we want a corresponding Operation or mlir::Value
  llvm::DenseMap<heavy::Binding*, mlir::Value> BindingTable;
  mlir::OpBuilder Builder;

public:
  bool IsTopLevel = false;

  explicit OpGen(heavy::Context& C)
    : Context(C),
      Builder(&(C.MlirContext))
  { }

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

  mlir::Value createTopLevelDefine(Symbol* S, Value *V, Value* OrigCall) {
    heavy::Value* EnvStack = Context.EnvStack;
    heavy::SourceLocation DefineLoc = OrigCall->getSourceLocation();
    // A module at the top of the EnvStack is mutable
    Module* M = nullptr;
    Value* EnvRest = nullptr;
    if (isa<Pair>(EnvStack)) {
      Value* EnvTop  = cast<Pair>(EnvStack)->Car;
      EnvRest = cast<Pair>(EnvStack)->Cdr;
      M = dyn_cast<Module>(EnvTop);
    }
    if (!M) return SetError("define used in immutable environment", OrigCall);

    mlir::Value InitVal = Visit(V);
    // If the name already exists in the current module
    // then it behaves like `set!`
    Binding* B = M->Lookup(S);
    if (B) {
      return SetError("TODO create SetOp for top level define", OrigCall);
      //B->Val = V;
      //return B;
    }

    // Top Level definitions may not shadow names in
    // the parent environment
    B = Context.Lookup(S, EnvRest);
    if (B) return SetError("define overwrites immutable location", S);

    // Create the BindingOp
    B = Context.CreateBinding(S, Context.CreateUndefined());
    mlir::Value BVal = create<BindingOp>(
        S->getSourceLocation(),
        InitVal);
    M->Insert(B);
    BindingTable.try_emplace(B, BVal);

    return create<DefineOp>(DefineLoc, BVal);
  }

  mlir::Value SetError(StringRef Str, Value* V) {
    Context.SetError(Str, V);
    return create<LiteralOp>(V->getSourceLocation(),
                             Context.CreateUndefined());
  }

private:
  mlir::Value VisitValue(Value* V) {
    return create<LiteralOp>(V->getSourceLocation(), V);
  }

  mlir::Value VisitSymbol(Symbol* S) {
    Binding* B = Context.Lookup(S);
    mlir::Value V = BindingTable.lookup(B);
    // V should be a value for a BindingOp or nothing
    // BindingOps are created in the `define` syntax
    if (V) return V;

    return SetError("unbound symbol: ", S);
  }

  mlir::Value HandleCall(Pair* P) {
    mlir::Value Fn = Visit(P->Car);
    llvm::SmallVector<mlir::Value, 16> Args;
    HandleCallArgs(P->Cdr, Args);
    return create<ApplyOp>(P->getSourceLocation(), Fn, Args);
  }

  void HandleCallArgs(Value *V,
                      llvm::SmallVectorImpl<mlir::Value>& Args) {
    if (isa<Empty>(V)) return;
    if (!isa<Pair>(V)) {
      SetError("improper list as call expression", V);
      return;
    }
    Pair* P = cast<Pair>(V);
    Args.push_back(Visit(P->Car));
    HandleCallArgs(P->Cdr, Args);
  }

  mlir::Value VisitPair(Pair* P) {
    Value* Operator = P->Car;
    // A named operator might point to some kind of syntax transformer
    if (Binding* B = Context.Lookup(Operator)) {
      Operator = B->getValue();
    }

    switch (Operator->getKind()) {
      case Value::Kind::BuiltinSyntax: {
        BuiltinSyntax* BS = cast<BuiltinSyntax>(Operator);
        return BS->Fn(*this, P);
      }
      case Value::Kind::Syntax:
        llvm_unreachable("TODO");
        return mlir::Value();
      default:
        IsTopLevel = false;
        return HandleCall(P);
    }
  }

#if 0 // TODO VectorOp??
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
#endif
};

mlir::Value opGen(Context& C, Value* V) {
  OpGen O(C);
  return O.Visit(V);
}

namespace {
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

void LoadBuiltinSyntax(Context& C) {
  C.AddBuiltinSyntax("quote",       builtin_syntax::quote);
  C.AddBuiltinSyntax("quasiquote",  builtin_syntax::quasiquote);
  C.AddBuiltinSyntax("define",      builtin_syntax::define);
}

