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
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;


OpGen::OpGen(heavy::Context& C)
  : Context(C),
    Builder(&(C.MlirContext)),
    LocalInits(&(C.MlirContext)),
    BindingTable(),
    BindingTableTop(BindingTable)
{
  mlir::ModuleOp TL =  Builder.create<mlir::ModuleOp>(
      Builder.getUnknownLoc());
  Builder.setInsertionPointToStart(TL.getBody());
  TopLevel = TL;
}

mlir::ModuleOp OpGen::getTopLevel() {
  return cast<mlir::ModuleOp>(TopLevel);
}

mlir::Value OpGen::createUndefined() {
  return create<UndefinedOp>(SourceLocation());
}

mlir::Value OpGen::createLambda(Value* Formals, Value* Body,
                                SourceLocation Loc,
                                llvm::StringRef Name) {
  BindingScope BScope(BindingTable);

  bool HasRestParam = false;
  heavy::EnvFrame* EnvFrame = Context.PushLambdaFormals(Formals,
                                                        HasRestParam);
  if (!EnvFrame) return Error();
  unsigned Arity = EnvFrame->getBindings().size();

  LambdaOp L = create<LambdaOp>(Loc, Name, Arity, HasRestParam,
                                /*Captures=*/llvm::ArrayRef<mlir::Value>{});

  mlir::Block& Block = *L.addEntryBlock();
  mlir::OpBuilder::InsertionGuard IG_1(Builder);
  Builder.setInsertionPointToStart(&Block);

  // Create the BindingOps for the arguments
  for (auto tup : llvm::zip(EnvFrame->getBindings(),
                            Block.getArguments())) {
    Binding *B        = std::get<0>(tup);
    mlir::Value Arg   = std::get<1>(tup);
    createBinding(B, Arg);
  }

  processBody(Loc, Body);
  Context.PopEnvFrame();

  return L;
}

bool OpGen::isLocalDefineAllowed() {
  mlir::Block* Block = Builder.getInsertionBlock();
  if (!Block) return false;
  return (Block->empty() ||
          isa<BindingOp>(Block->back()));
}

void OpGen::processBody(SourceLocation Loc, Value* Body) {
  mlir::OpBuilder::InsertionGuard IG(LocalInits);
  LocalInits = Builder;

  // Each local "define" should update the LocalInits
  // insertion point

  IsTopLevel = false;

  if (!isa<Pair>(Body)) {
    SetError(Loc, "body must contain an expression", Body);
  }

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

  {
    mlir::Block* Block = LocalInits.getInsertionBlock();
    mlir::Block::iterator Itr = LocalInits.getInsertionPoint();
    if (Block && Itr != Block->end() && isa<BindingOp>(*Itr)) {
      BindingOp LastBindingOp = cast<BindingOp>(*Itr);
      LocalInits.setInsertionPointAfter(LastBindingOp);
    }
  }
  mlir::OpBuilder::InsertionGuard IG2(Builder);
  Builder = LocalInits;

  Value* Env = Context.EnvStack;
  while (Pair* EnvPair = dyn_cast<Pair>(Env)) {
    Binding* B = dyn_cast<Binding>(EnvPair->Car);
    // We should eventually hit the EnvFrame that wraps this local scope
    if (!B) break;
    mlir::Value BVal = BindingTable.lookup(B);
    mlir::Value Init = VisitDefineArgs(B->getValue());
    SourceLocation Loc = B->getValue()->getSourceLocation();
    assert(BVal && "BindingTable should have an entry for local define");

    create<SetOp>(Loc, BVal, Init);
    Env = EnvPair->Cdr;
  }
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  SourceLocation SymbolLoc = B->getName()->getSourceLocation();
  mlir::Value BVal = create<BindingOp>(SymbolLoc, Init);
  BindingTable.insert(B, BVal);

  return BVal;
}

mlir::Value OpGen::createDefine(Symbol* S, Value* DefineArgs,
                                           Value* OrigCall) {
  if (isTopLevel()) return createTopLevelDefine(S, DefineArgs, OrigCall);
  if (!isLocalDefineAllowed()) return SetError("unexpected define", OrigCall);
  // create the binding with a lazy init
  // (include everything after the define
  //  keyword to visit it later because it could
  //  be a terse lambda syntax)
  Binding* B = Context.CreateBinding(S, DefineArgs);
  // push to the local environment
  Context.PushLocalBinding(B);
  mlir::Value BVal = createBinding(B, createUndefined());
  // Have LocalInits insertion point to the last BindingOp
  LocalInits.setInsertionPoint(BVal.getDefiningOp());
  assert(isLocalDefineAllowed() && "define should still be allowed");
  return BVal;
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, Value* DefineArgs,
                                        Value* OrigCall) {
  mlir::Value Init = VisitDefineArgs(DefineArgs);
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
    SourceLocation DefineLoc = OrigCall->getSourceLocation();
    mlir::Value BVal = BindingTable.lookup(B);
    assert(BVal && "expecting BindingOp for Binding");
    return create<SetOp>(DefineLoc, BVal, Init);
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

// This handles everything after the `define` keyword
// including terse lambda syntax. This supports lazy
// visitation of local bindings' initializers.
mlir::Value OpGen::VisitDefineArgs(Value* Args) {
  Pair* P = cast<Pair>(Args);
  if (Pair* LambdaSpec = dyn_cast<Pair>(P->Car)) {
    // we already checked the name
    Symbol* S = cast<Symbol>(LambdaSpec->Car);
    Value* Formals = LambdaSpec->Cdr;
    Value* Body = P->Cdr;
    return createLambda(Formals, Body,
                        S->getSourceLocation(),
                        S->getVal());

  }
  if (isa<Symbol>(P->Car) && isa<Pair>(P->Cdr)) {
    return Visit(cast<Pair>(P->Cdr)->Car);
  }
  return SetError("invalid define syntax", P);
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
