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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using namespace heavy;

class OpGen : public ValueVisitor<OpGen, mlir::Value*> {
  friend class ValueVisitor<OpGen, Value*>;
  heavy::Context& Context;
  // heavy::Context::Lookup returns a heavy::Binding that
  // we want a corresponding Operation or mlir::Value
  llvm::DenseMap<heavy::Binding*, mlir::Value> BindingTable;
  mlir::OpBuilder Builder;
  bool IsTopLevel = false;

public:
  OpGen(heavy::Context& C)
    : Context(C),
      Builder(&(C.MlirContext))
  { }

  mlir::Value VisitTopLevel(Value* V) {
    IsTopLevel = true;
    return Visit(V);
  }

  template <typename Op, typename ...Args>
  mlir::Value create(Args&& ...args) {
    return Builder.create<Op>(
      std::forward<Args>(args)...);
  }

  mlir::Value createTopLevelDefine(Symbol* S, Value *V, Value* OrigCall) {
    // A module at the top of the EnvStack is mutable
    Module* M = nullptr;
    Value* EnvRest = nullptr;
    if (isa<Pair>(EnvStack)) {
      Value* EnvTop  = cast<Pair>(EnvStack)->Car;
      EnvRest = cast<Pair>(EnvStack)->Cdr;
      M = dyn_cast<Module>(EnvTop);
    }
    if (!M) return SetError("define used in immutable environment", OrigCall);

    // If the name already exists in the current module
    // then it behaves like `set!`
    Binding* B = M->Lookup(S);
    if (B) {
      return SetError("TODO create SetOp for top level define");
      //B->Val = V;
      //return B;
    }

    // Top Level definitions may not shadow names in
    // the parent environment
    B = Lookup(S, EnvRest);
    if (B) return SetError("define overwrites immutable location", S);

    // Create the BindingOp
    B = Context.CreateBinding(S, V);
    mlir::Value B_op = create<BindingOp>(Context.CreateBinding(S, V));
    M->Insert(B);
    BindingTable.try_emplace(B, B_op);

    return create<DefineOp>(B_Op);
  }

  mlir::Value SetError(StringRef Str, Value* V) {
    String* Msg = Context.CreateString("unbound symbol: ", Str);
    Context.SetError(Msg, S);
    return Builder.create<LiteralOp>(Context.CreateUndefined());
  }

private:
  mlir::Value VisitValue(Value* V) {
    return Builder.create<LiteralOp>(V);
  }

  mlir::Value VisitSymbol(Symbol* S) {
    Binding* B = Context.Lookup(S)
    mlir::Value V = BindingTable.lookup(B);
    // V should be a value for a BindingOp or nothing
    // BindingOps are created in the `define` syntax
    if (V) return V;

    return SetError("unbound symbol: ", S->getVal());
  }

  mlir::Value HandleCall(Pair* P) {
    mlir::Value Fn = Visit(P->Car);
    llvm::SmallVector<mlir::Value, 16> Args;
    HandleCallArgs(P->Cdr, Args);
    return Builder.create<ApplyOp>(Fn, Args);
  }

  void HandleCallArgs(Value *V,
                      llvm::SmallVectorImpl<mlir::Value>& Args) {
    if (isa<Empty>(V)) return V;
    if (!isa<Pair>(V)) {
      return SetError("improper list as call expression", V);
    }
    Pair* P = cast<Pair>(V);
    Args.push_back(Visit(P->Car));
    HandleCallArgs(P->Cdr, Args);
  }

  mlir::Value VisitPair(Pair* P) {
    if (Context.CheckError()) return Context.CreateEmpty();
    Value* Operator = P->Car;
    // A named operator might point to some kind of syntax transformer
    if (Binding* B = Context.Lookup(Operator)) {
      Operator = B->getValue();
    }

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


mlir::Value opGen(Context&, Value* V, Value* EnvStack) {
  OpGen O(C, EnvStack);
  return O.Visit(V);
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
    // TODO build a DefineOp and add it to the map of bindings to ops
    Binding* B = C.CreateGlobal(S, V, P);
    return OG.create<DefineOp>(
    return C.CreatePair(C.GetBuiltin("__define"),
            C.CreatePair(C.CreateQuote(
              C.CreateGlobal(S, V, P))));
  } else {
    // Handle internal definitions inside
    // lambda syntax
    return OG.SetError("unexpected define", P);
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
