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
    Builder(&(C.MlirContext))
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

  // Create the BindingOps for the arguments
  llvm::SmallVector<mlir::Value, 8> BindingOps;
  for (Binding* B : EnvFrame->getBindings()) {
    mlir::Value Arg = /* TODO the value of the arg */;
    mlir::Value BVal = createBinding(B, Arg);
    BindingOps.push_back(BVal);
  }

  // This will be a SequenceOp or single tail Op
  mlir::Operation* BodyOp = createBody(Block, Body);

  //
  LambdaOp L = create<LambdaOp>(Loc, Name, Arity, HasRestParam,
                                /*Captures=*/llvm::ArrayRef<mlir::Value>{});
  mlir::Block& Block = *L.addEntryBlock();
  // Add the BindingOps
  for (mlir::Value BVal : BindingOps) {
    assert(BVal->getDefiningOp() && "Expecting an Operation");
    Block.push_back(BVal->getDefiningOp());
  }
  // Create/Add the BindingOps from the Bindings in the EnvStack (local defines)
  //            Pop them as we go
  // Create/Add the corresponding SetOps to lazy initialize the local defines
  // Pop the EnvFrame
  return L;
}

void processBody(mlir::Block* Block, Value* Body) {
  // TODO we need to splice nested sequences

  // A "Body" consists of local defines and a sequence.
  // The sequence is either a single op or we wrap multiple
  // ops in a SequenceOp
  Value* RestBody = Body;
  // enumerate local defines until we get an op
  mlir::Value FirstOp;
  while (Pair* Current = dyn_cast<Pair>(RestBody)) {
    FirstOp = Visit(Current->Car);
    RestBody = Current->Cdr;
    if (FirstOp) break;
  }
  // insert the BindingOps
  for (std::pair<Symbol*, Value*> X : LocalDefines) {
    mlir::Value BVal = createBinding(Context.CreateBinding(X.first));
    Block.push_back(BVal);
  }
  LocalDefines.clear();

  // insert the DefineOps (which initialize the Bindings)
  // Walk the EnvStack while we get Bindings
  for (Value* InitExpr : InitExprs) {
    mlir::Value Init = Visit(InitExpr);
    mlir::Value DVal = create<DefineOp>(DefineLoc, BVal, Init);
    Block.push_back(DVal);
  }

  assert(FirstOp && "A body must have at least one expression");
  // A single expression can be added and we're done
  if (!isa<Pair>(RestBody)) {
    Block.push_back(FirstOp);
    return L;
  }

  llvm::SmallVector<mlir::Value, 16> Ops;
  Ops.push_back(FirstOp);
  mlir::Value SeqOp = createSequence(RestBody->getSourceLocation(),
                                     RestBody, Ops);
  Block.push_back(SeqOp);
}

mlir::Value OpGen::createSequence(SourceLocation Loc, Value* Exprs) {
  llvm::SmallVector<mlir::Value, 16> Ops;
  return createSequence(Loc, Exprs, Ops);
}

mlir::Value OpGen::createSequence(SourceLocation Loc, Value* Exprs,
                              llvm::SmallVectorImpl<mlir::Value>& Ops) {
  while (Pair* Current = dyn_cast<Pair>(Exprs)) {
    Exprs = Current->Cdr;
    Ops.push_back(Visit(Current->Car));
  }
  create<SequenceOp>(Loc, Ops);
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  if (!Init) Init = createUndefined();
  Sourcelocation SymbolLoc = B->getSymbol()->getSourceLocation();
  mlir::Value BVal = create<BindingOp>(SymbolLoc, Init);
  BindingTable.try_emplace(B, BVal);

  return BVal;
}

// This could be a top-level or local variable
// BindingTable scope should already be handled
mlir::Value OpGen::createDefine(Binding* B, mlir::Value Init,
                                SourceLocation DefineLoc) {
  mlir::Value BVal = createBinding(B, Init);
  mlir::Value DVal = create<DefineOp>(DefineLoc, BVal);

  return DVal;
}

mlir::Value OpGen::createDefine(Symbol* S, Value* V, Value* OrigCall) {
  mlir::Value Init = Visit(V);
  if (OG.IsTopLevel) {
    return createTopLevelDefine(S, Init, OrigCall);
  }
  return createDefine(S, Init, OrigCall);
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, mlir::Value Init,
                                        Value* OrigCall) {
  heavy::Value* EnvStack = Context.EnvStack;
  heavy::SourceLocation DefineLoc = OrigCall->getSourceLocation();
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
  if (B) {
    // require IsTopLevel == true
    return SetError("TODO create SetOp for top level define", OrigCall);
  }

  // Top Level definitions may not shadow names in
  // the parent environment
  B = Context.Lookup(S, EnvRest);
  if (B) return SetError("define overwrites immutable location", S);

  return createTopLevelDefine(S, Init, M, DefineLoc);
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, mlir::Value Init,
                                Module* M, SourceLocation DefineLoc) {
  Binding* B = Context.CreateBinding(S, Context.CreateUndefined());
  M->Insert(B);
  return createDefine(B, Init, DefineLoc);
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
