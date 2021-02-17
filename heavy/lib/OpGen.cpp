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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;


OpGen::OpGen(heavy::Context& C)
  : Context(C),
    Builder(&(C.MlirContext)),
    LocalInits(&(C.MlirContext))
{ }

mlir::Value OpGen::createLambda(Value* Formals, Value* Body,
                                SourceLocation Loc,
                                llvm::StringRef Name) {
  auto BScope = BindingScope(BindingTable);

  bool HasRestParam = false;
  heavy::EnvFrame* EnvFrame = Context.PushLambdaFormals(Formals,
                                                        HasRestParam);
  if (!E) return Error(Loc);
  unsigned Arity = E->getBindings().size();

  LambdaOp L = create<LambdaOp>(Loc, Name, Arity, HasRestParam,
                                /*Captures=*/llvm::ArrayRef<mlir::Value>{});

  mlir::Block& Block = *L.addEntryBlock();
  mlir::InsertionGuard IG_1(Builder);
  mlir::InsertionGuard IG_2(LocalInits);

  Builder.setInsertionPointToStart(Block);

  // Create the BindingOps for the arguments
  for (auto tup : llvm::zip(EnvFrame->getBindings(),
                            Block->getArguments())) {
    Binding *B        = std::get<0>(tup);
    mlir::Value Arg   = std::get<1>(tup);
    createBinding(B, Arg);
  }

  processBody(Body);
  Context.PopEnvFrame();

  return L;
}

bool OpGen::isLocalDefineAllowed() {
  return LocalInits.getInsertionPoint() == Builder.getInsertionPoint();
}

void OpGen::processBody(Value* Body) {
  // InsertionGuards for Builder and LocalInits
  // should be set above this
  mlir::InsertionGuard IG(LocalInits);
  LocalInits = Builder;

  // Each local "define" should update the LocalInits
  // insertion point

  IsTopLevel = false;

  Value* RestBody = Body;
  mlir::Value LastOp;
  while (Pair* Current = dyn_cast<Pair>(RestBody)) {
    LastOp = Visit(Current->Car);
    RestBody = Current->Cdr;
  }

  // The BindingOps for the local defines have
  // been inserted by the `define` syntax. They are
  // initialized to "undefined" and their corresponding
  // heavy::Bindings placed in the EnvStack.
  // Walk the EnvStack to collect these and insert the lazy
  // initializers via SetOp
  Value* Env = Context.EnvStack;
  while (Pair* EnvPair : dyn_cast<Pair>(Env)) {
    Binding* B = dyn_cast<Binding>(EnvPair.Car);
    // We should eventually hit the EnvFrame that wraps this local scope
    if (!B) break;
    mlir::Value BVal  = BindingTable.lookup(B);
    assert(BVal && "BindingTable should have an entry for local define");
    mlir::Value Init  = Visit(B.getValue())
    create<SetOp>(LocalInits, BVal, Init);
  }
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  Sourcelocation SymbolLoc = B->getSymbol()->getSourceLocation();
  mlir::Value BVal = create<BindingOp>(SymbolLoc, Init);
  BindingTable.try_emplace(B, BVal);

  return BVal;
}

mlir::Value OpGen::createDefine(Symbol* S, Value* V, Value* OrigCall) {
  mlir::Value Init = Visit(V);
  if (OG.isTopLevel()) {
    return createTopLevelDefine(S, Init, OrigCall);
  } else if (OG.isLocalDefineAllowed()) {
    // create the binding with a lazy init
    Binding* B = Context.CreateBinding(S, V);
    // push to the local environment
    Context.PushLocalBinding(B);
    mlir::Value BVal = createBinding(B, createUndefined());
    // The LocalInits insertion point should be
    // after the last "define"
    LocalInits.setInsertionPointAfter(BVal->getDefiningOp());
    assert(isLocalDefineAllowed() && "define should still be allowed");
    return BVal;
  }

  return SetError("unexpected define", OrigCall);
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, mlir::Value Init,
                                        Value* OrigCall) {
  heavy::Value* EnvStack = Context.EnvStack;

  // A module at the top of the EnvStack is mutable
  Module* M = nullptr;
  Value* EnvRest = nullptr;
  if (isa<Pair>(EnvStack)) {
    Value* EnvTop = cast<Pair>(EnvStack)->Car;
    EnvRest = cast<Pair>(EnvStack)->Cdr;
    M = dyn_cast<Module>(EnvTop);
  }
  if (!M) return SetError("define used in immutable environment", OrigCall);

  // If the name already exists in the current module
  // then it behaves like `set!`
  Binding* B = M->Lookup(S);
  if (B && !B->isSyntactic()) {
    return create<SetOp>(BVal, Init);
  }

  // Top Level definitions may not shadow names in
  // the parent environment
  B = Context.Lookup(S, EnvRest);
  if (B) {
    return SetError("define overwrites immutable location", S);
  }

  return createTopLevelDefine(S, Init, M);
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, mlir::Value Init,
                                        Module* M) {
  Binding* B = Context.CreateBinding(S, Context.CreateUndefined());
  M->Insert(B);
  return createBinding(B, Init);
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
    default: {
      IsTopLevel = false;
      return HandleCall(P);
    }
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
