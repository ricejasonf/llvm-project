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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;

OpGen::OpGen(heavy::Context& C)
  : Context(C),
    Builder(&(C.MlirContext)),
    LocalInits(&(C.MlirContext)),
    BindingTable()
{
  mlir::ModuleOp TL =  Builder.create<mlir::ModuleOp>(
      Builder.getUnknownLoc());
  Builder.setInsertionPointToStart(TL.getBody());
  TopLevel = TL;
  LambdaScopes.emplace_back(TL, BindingTable);
}

mlir::ModuleOp OpGen::getTopLevel() {
  return cast<mlir::ModuleOp>(TopLevel);
}

mlir::Value OpGen::createUndefined() {
  return create<UndefinedOp>(SourceLocation());
}

mlir::FunctionType OpGen::createFunctionType(unsigned Arity,
                                             bool HasRestParam) {
  mlir::Type HeavyLambdaTy  = Builder.getType<HeavyLambda>();
  mlir::Type HeavyRestTy    = Builder.getType<HeavyRest>();
  mlir::Type HeavyValueTy   = Builder.getType<HeavyValue>();

  llvm::SmallVector<mlir::Type, 16> Types{};
  Types.push_back(HeavyLambdaTy);
  if (Arity > 0) {
    for (unsigned i = 0; i < Arity - 1; i++) {
      Types.push_back(HeavyValueTy);
    }
    mlir::Type LastParamTy = HasRestParam ? HeavyValueTy :
                                            HeavyRestTy;
    Types.push_back(LastParamTy);
  }

  return Builder.getFunctionType(Types, HeavyValueTy);
}

mlir::Value OpGen::createLambda(Value Formals, Value Body,
                                SourceLocation Loc,
                                llvm::StringRef Name) {
  IsTopLevel = false;

  std::string GeneratedName;
  if (Name.size() == 0) {
    GeneratedName = generateLambdaName();
    Name = GeneratedName;
  }

  bool HasRestParam = false;
  heavy::EnvFrame* EnvFrame = Context.PushLambdaFormals(Formals,
                                                        HasRestParam);
  if (!EnvFrame) return Error();
  unsigned Arity = EnvFrame->getBindings().size();
  mlir::FunctionType FT = createFunctionType(Arity, HasRestParam);

  auto F = create<mlir::FuncOp>(Loc, Name, FT);
  LambdaScope LScope(*this, F);

  // Insert into the function body
  {
    mlir::OpBuilder::InsertionGuard IG(Builder);
    mlir::Block& Block = *F.addEntryBlock();
    Builder.setInsertionPointToStart(&Block);

    // ValueArgs drops the Closure arg at the front
    auto ValueArgs  = Block.getArguments().drop_front();
    // Create the BindingOps for the arguments
    for (auto tup : llvm::zip(EnvFrame->getBindings(),
                              ValueArgs)) {
      Binding *B        = std::get<0>(tup);
      mlir::Value Arg   = std::get<1>(tup);
      createBinding(B, Arg);
    }

    mlir::Value BodyVal = createBody(Loc, Body);

    // Create terminator
    Builder.create<ContOp>(BodyVal.getLoc(), BodyVal);

    Context.PopEnvFrame();
  }

  LambdaScopeNode& LS = LambdaScopes.back();
  return create<LambdaOp>(Loc, Name, LS.Captures);
}

bool OpGen::isLocalDefineAllowed() {
  mlir::Block* Block = Builder.getInsertionBlock();
  if (!Block) return false;
  return (Block->empty() ||
          isa<BindingOp>(Block->back()));
}

// processSequence creates a sequence of operations in the current block
mlir::Value OpGen::createSequence(SourceLocation Loc, Value Body) {
  if (!isa<Pair>(Body)) {
    SetError(Loc, "sequence must contain an expression", Body);
  }

  Value Rest = Body;
  {
    TailPosScope TPS(*this);
    IsTailPos = false;
    while (true) {
      Pair* Current = cast<Pair>(Rest);
      Rest = Current->Cdr;
      if (isa<Empty>(Rest)) {
        Rest= Current->Car;
        break;
      }
      Visit(Current->Car);
    }
  }
  // This could be in tail position
  return Visit(Rest);
}

mlir::Value OpGen::createBody(SourceLocation Loc, Value Body) {
  mlir::OpBuilder::InsertionGuard IG(LocalInits);
  LocalInits = Builder;

  // Each local "define" should update the LocalInits
  // insertion point

  IsTopLevel = false;
  mlir::Value Result = createSequence(Loc, Body);

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
  {
    mlir::OpBuilder::InsertionGuard IG2(Builder);
    Builder = LocalInits;

    Value Env = Context.EnvStack;
    while (Pair* EnvPair = dyn_cast<Pair>(Env)) {
      Binding* B = dyn_cast<Binding>(EnvPair->Car);
      // We should eventually hit the EnvFrame that wraps this local scope
      if (!B) break;
      mlir::Value BVal = BindingTable.lookup(B);
      mlir::Value Init = VisitDefineArgs(B->getValue());
      SourceLocation Loc = B->getValue().getSourceLocation();
      assert(BVal && "BindingTable should have an entry for local define");

      create<SetOp>(Loc, BVal, Init);
      Env = EnvPair->Cdr;
    }
    // Builder is restored to the end of the body block
  }

  return Result;
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  SourceLocation SymbolLoc = B->getName()->getSourceLocation();
  mlir::Value BVal = create<BindingOp>(SymbolLoc, Init, isTopLevel());
  BindingTable.insert(B, BVal);

  if (isTopLevel()) {
    BVal.getDefiningOp<BindingOp>().setName(B->getName()->getVal());
  }

  return BVal;
}

mlir::Value OpGen::createDefine(Symbol* S, Value DefineArgs,
                                           Value OrigCall) {
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

mlir::Value OpGen::createTopLevelDefine(Symbol* S, Value DefineArgs,
                                        Value OrigCall) {
  SourceLocation DefineLoc = OrigCall.getSourceLocation();
  heavy::Value EnvStack = Context.EnvStack;

  // A module at the top of the EnvStack is mutable
  Module* M = nullptr;
  Value EnvRest = nullptr;
  if (isa<Pair>(EnvStack)) {
    Value EnvTop = cast<Pair>(EnvStack)->Car;
    EnvRest = cast<Pair>(EnvStack)->Cdr;
    M = dyn_cast<Module>(EnvTop);
  }
  if (!M) return SetError("define used in immutable environment", OrigCall);

  // If the name already exists in the current module
  // then it behaves like `set!`
  Binding* B = M->Lookup(S);
  if (B && !B->isSyntactic()) {
    mlir::Value BVal = BindingTable.lookup(B);
    assert(BVal && "expecting BindingOp for Binding");
    mlir::Value Init = VisitDefineArgs(DefineArgs);
    return create<SetOp>(DefineLoc, BVal, Init);
  }

  // Top Level definitions may not shadow names in
  // the parent environment
  B = Context.Lookup(S, EnvRest);
  if (B) {
    return SetError("define overwrites immutable location", S);
  }

  B = Context.CreateBinding(S, DefineArgs);
  M->Insert(B);
  mlir::Value BVal = createBinding(B, createUndefined());
  mlir::Value Init = VisitDefineArgs(DefineArgs);
  return create<SetOp>(DefineLoc, BVal, Init);
}

// used by builtins
mlir::Value OpGen::createTopLevelDefine(Symbol* S, mlir::Value Init,
                                        Module* M) {
  Binding* B = Context.CreateBinding(S, Context.CreateUndefined());
  M->Insert(B);
  return createBinding(B, Init);
}

mlir::Value OpGen::createIf(SourceLocation Loc, Value Cond, Value Then,
                            Value Else) {
  // Cond
  mlir::Value CondResult;
  {
    TailPosScope TPS(*this);
    IsTailPos = false;
    CondResult = Visit(Cond);
  }

  auto IfOp = create<heavy::IfOp>(Loc, CondResult);
  IfOp.thenRegion().push_back(new mlir::Block());
  IfOp.elseRegion().push_back(new mlir::Block());

  // Then
  {
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&IfOp.thenRegion().front());
    mlir::Value Result = Visit(Then);
    create<ContOp>(Loc, Result);
  }

  // Else
  {
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&IfOp.elseRegion().front());
    mlir::Value Result = Visit(Else);
    create<ContOp>(Loc, Result);
  }

  return IfOp;
}

// LHS can be a symbol or a binding
mlir::Value OpGen::createSet(SourceLocation Loc, Value LHS,
                                                 Value RHS) {
  assert((isa<Binding>(LHS) || isa<Symbol>(LHS)) &&
      "expects a Symbol or Binding for LHS");
  mlir::Value BVal = Visit(LHS);
  mlir::Value ExprVal = Visit(RHS);
  return create<SetOp>(Loc, BVal, ExprVal);
}

// This handles everything after the `define` keyword
// including terse lambda syntax. This supports lazy
// visitation of local bindings' initializers.
mlir::Value OpGen::VisitDefineArgs(Value Args) {
  Pair* P = cast<Pair>(Args);
  if (Pair* LambdaSpec = dyn_cast<Pair>(P->Car)) {
    // we already checked the name
    Symbol* S = cast<Symbol>(LambdaSpec->Car);
    Value Formals = LambdaSpec->Cdr;
    Value Body = P->Cdr;
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
  return VisitBinding(B);
}

mlir::Value OpGen::VisitBinding(Binding* B) {
  mlir::Value V = BindingTable.lookup(B);
  // V should be a value for a BindingOp or nothing
  // BindingOps are created in the `define` syntax

  assert(V && "binding must exist in BindingTable");
  return LocalizeValue(B, V);
}

mlir::Value OpGen::HandleCall(Pair* P) {
  bool TailPos = isTailPos();
  IsTopLevel = false;
  mlir::Value Fn = Visit(P->Car);
  llvm::SmallVector<mlir::Value, 16> Args;
  HandleCallArgs(P->Cdr, Args);
  return create<ApplyOp>(P->getSourceLocation(), Fn, Args,
                         TailPos);
}

void OpGen::HandleCallArgs(Value V,
                    llvm::SmallVectorImpl<mlir::Value>& Args) {
  if (isa<Empty>(V)) return;
  if (!isa<Pair>(V)) {
    SetError("improper list as call expression", V);
    return;
  }
  // FIXME this would probably be
  // better as a loop
  Pair* P = cast<Pair>(V);
  {
    TailPosScope TPS(*this);
    IsTailPos = false;
    Args.push_back(Visit(P->Car));
  }
  HandleCallArgs(P->Cdr, Args);
}

mlir::Value OpGen::VisitPair(Pair* P) {
  Value Operator = P->Car;
  // A named operator might point to some kind of syntax transformer
  if (Binding* B = Context.Lookup(Operator)) {
    Operator = B->getValue();
  }

  switch (Operator.getKind()) {
    case ValueKind::BuiltinSyntax: {
      BuiltinSyntax* BS = cast<BuiltinSyntax>(Operator);
      return BS->Fn(*this, P);
    }
    case ValueKind::Syntax:
      llvm_unreachable("TODO");
      return mlir::Value();
    default: {
      return HandleCall(P);
    }
  }
}

#if 0 // TODO VectorOp??
mlir::Value OpGen::VisitVector(Vector* V) {
  llvm::ArrayRef<Value> Xs = V->getElements();
  Vector* New = Context.CreateVector(Xs.size());
  llvm::MutableArrayRef<Value> Ys = New->getElements();
  for (unsigned i = 0; i < Xs.size(); ++i) {
    Visit(Xs[i]);
    Ys[i] = Visit(Xs[i]);
  }
  return New;
}
#endif

// LocalizeValue - If a value belongs to a parent region from
//                 which the current region should be isolated
//                 we generate one of two operations (LoadGlobal or
//                 LoadCapture), and we return the result of that.
//                 Tracking of captures for nested scopes is handled
//                 here too.
//
mlir::Value OpGen::LocalizeValue(heavy::Binding* B, mlir::Value V) {
  mlir::Operation* Op = V.getDefiningOp();
  assert(Op && "value should be an operation result");

  mlir::Operation* Owner = Op
    ->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();

  return LocalizeRec(B, Op, Owner, LambdaScopes.rbegin());
}

mlir::Value OpGen::LocalizeRec(heavy::Binding* B,
                               mlir::Operation* Op,
                               mlir::Operation* Owner,
                               LambdaScopeIterator Itr) {
  assert(Itr != LambdaScopes.rend() && "value must be in a scope");
  LambdaScopeNode& LS = *Itr;
  if (LS.Op == Owner) return Op->getResult(0);

  heavy::SourceLocation Loc = {};
  llvm::StringRef SymName = mlir::SymbolTable::getSymbolAttrName();
  mlir::Value NewVal;

  if (auto S = Op->getAttrOfType<mlir::StringAttr>(SymName)) {
    NewVal = create<LoadGlobalOp>(Loc, S.getValue());
  } else {
    mlir::Value ParentLocal = LocalizeRec(B, Op, Owner, ++Itr);

    mlir::OpBuilder::InsertionGuard IG(Builder);
    auto FuncOp = cast<mlir::FuncOp>(LS.Op);
    mlir::Block& Block = FuncOp.getBody().front();
    Builder.setInsertionPointToStart(&Block);

    LS.Captures.push_back(ParentLocal);
    uint32_t Index = LS.Captures.size() - 1;
    mlir::Value Closure = Block.getArguments().front();
    NewVal = create<LoadClosureOp>(Loc, Closure, Index);
  }

  BindingTable.insertIntoScope(&LS.BindingScope_, B, NewVal);
  return NewVal;
}

Value heavy::eval(Context& C, Value V, Value EnvStack) {
  heavy::Value Args[2] = {V, EnvStack};
  int ArgCount = EnvStack ? 2 : 1;
  return builtin::eval(C, ValueRefs(Args, ArgCount));
}
