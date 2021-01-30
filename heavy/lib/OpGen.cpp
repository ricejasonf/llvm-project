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

#include "heavy/Builtins.h"
#include "heavy/Dialect.h"
#include "heavy/HeavyScheme.h"
#include "heavy/OpGen.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;


OpGen::OpGen(heavy::Context& C)
  : Context(C),
    Builder(&(C.MlirContext))
{
  // Load Builtin Syntax
  C.AddBuiltinSyntax("quote",       builtin_syntax::quote);
  C.AddBuiltinSyntax("quasiquote",  builtin_syntax::quasiquote);
  C.AddBuiltinSyntax("define",      builtin_syntax::define);

  // Load Builtin Procedures
  C.AddBuiltin("+",                 builtin::operator_add);
  C.AddBuiltin("*",                 builtin::operator_mul);
  C.AddBuiltin("-",                 builtin::operator_sub);
  C.AddBuiltin("/",                 builtin::operator_div);
  C.AddBuiltin("list",              builtin::list);
  C.AddBuiltin("append",            builtin::append);
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, Value *V, Value* OrigCall) {
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
  mlir::Value BVal = create<BindingOp>(S->getSourceLocation(), InitVal);
  mlir::Value DVal = create<DefineOp>(DefineLoc, BVal);
  M->Insert(B);
  BindingTable.try_emplace(B, BVal);
  assert(BindingTable.lookup(B) == BVal &&
      "BindingTable should contain the value");
  BindingTable.lookup(B).dump();

  return DVal;
}

mlir::Value OpGen::VisitSymbol(Symbol* S) {
  Binding* B = Context.Lookup(S);

  if (!B) {
    String* Msg = Context.CreateString("unbound symbol '",
                                       S->getVal(), "'");
    return SetError(Msg, S);
  }
  mlir::Value V = BindingTable.lookup(B);
  // V should be a value for a BindingOp or nothing
  // BindingOps are created in the `define` syntax
  if (V) return V;

  // FIXME this is happening for BuiltinFns
  String* Msg = Context.CreateString("binding has no associated value for '",
                                     S->getVal(), "'");
  return SetError(Msg, S);

}

mlir::Value OpGen::HandleCall(Pair* P) {
  mlir::Value Fn = Visit(P->Car);
  llvm::SmallVector<mlir::Value, 16> Args;
  HandleCallArgs(P->Cdr, Args);
  return create<ApplyOp>(P->getSourceLocation(), Fn, Args);
}

void OpGen::HandleCallArgs(Value *V,
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

mlir::Value OpGen::VisitPair(Pair* P) {
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
mlir::Value OpGen::VisitVector(Vector* V) {
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

Value* heavy::eval(Context& C, Value* V, Value* EnvStack) {
  heavy::Value* Args[2] = {V, EnvStack};
  int ArgCount = EnvStack ? 2 : 1;
  return builtin::eval(C, ValueRefs(Args, ArgCount));
}
