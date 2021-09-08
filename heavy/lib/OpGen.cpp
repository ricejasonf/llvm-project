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
#include "heavy/Context.h"
#include "heavy/Dialect.h"
#include "heavy/Mangle.h"
#include "heavy/OpGen.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;

namespace {
}

OpGen::OpGen(heavy::Context& C)
  : Context(C),
    TopLevelBuilder(&(C.MlirContext)),
    Builder(&(C.MlirContext)),
    LocalInits(&(C.MlirContext)),
    BindingTable()
{
  mlir::ModuleOp TL =  Builder.create<mlir::ModuleOp>(
      Builder.getUnknownLoc());
  TopLevelBuilder.setInsertionPointToStart(TL.getBody());
  TopLevel = TL;
  LambdaScopes.emplace_back(TL, BindingTable);
}

std::string OpGen::mangleFunctionName(llvm::StringRef Name) {
  heavy::Mangler Mangler(Context);
  if (Name.empty()) {
    return Mangler.mangleAnonymousId(getModulePrefix(), LambdaNameCount++);
  }
  return Mangler.mangleFunction(getModulePrefix(), Name);
}

mlir::ModuleOp OpGen::getTopLevel() {
  return cast<mlir::ModuleOp>(TopLevel);
}

mlir::Operation* OpGen::VisitTopLevel(Value V) {
  IsTopLevel = true;

  // We use a null Builder to denote that we should
  // insert into a lazily created CommandOp by default
  mlir::OpBuilder::InsertionGuard IG(Builder);
  Builder.clearInsertionPoint();

  Visit(V);
  mlir::Operation* Op = (TopLevelBuilder.getBlock()->back().getPrevNode());
  assert((isa<CommandOp, GlobalOp>(Op)) &&
      "top level operation must be CommandOp or GlobalOp");

  // FIXME this may require a less ad hoc solution
  if (heavy::CommandOp CommandOp = dyn_cast<heavy::CommandOp>(Op)) {
    mlir::Block& Block = CommandOp.body().front();
    assert(!Block.empty() && "command op must have body");
    if (!Block.back().isKnownTerminator()) {
      mlir::OpBuilder::InsertionGuard IG(Builder);
      Builder.setInsertionPointToEnd(&Block);
      create<ContOp>(heavy::SourceLocation(), createUndefined());
    }
  }

  // pop continuation scopes to the TopLevel
  while (LambdaScopes.size() > 0) {
    if (LambdaScopes.back().Op == TopLevel) break;
    PopContinuationScope();
  }
  return Op;
}

void OpGen::insertTopLevelCommandOp(SourceLocation Loc) {
  auto CommandOp = create<heavy::CommandOp>(TopLevelBuilder, Loc);
  mlir::Block& Block = *CommandOp.addEntryBlock();
  // overwrites Builder without reverting it
  Builder.setInsertionPointToStart(&Block);
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

  std::string MangledName = mangleFunctionName(Name);

  bool HasRestParam = false;
  heavy::EnvFrame* EnvFrame = Context.PushLambdaFormals(Formals,
                                                        HasRestParam);
  if (!EnvFrame) return Error();
  unsigned Arity = EnvFrame->getBindings().size();
  mlir::FunctionType FT = createFunctionType(Arity, HasRestParam);

  auto F = create<mlir::FuncOp>(Loc, MangledName, FT);
  LambdaScope LS(*this, F);

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

    // If Result is null then it already
    // has a terminator.
    if (mlir::Value Result = createBody(Loc, Body)) {
      Builder.create<ContOp>(Result.getLoc(), Result);
    }

    Context.PopEnvFrame();
  }

  return create<LambdaOp>(Loc, MangledName, LS.Node.Captures);
}

void OpGen::PopContinuationScope() {
  mlir::OpBuilder::InsertionGuard IG(Builder);
  LambdaScopeNode& LS = LambdaScopes.back();
  mlir::Location Loc = LS.Op->getLoc();
  Builder.setInsertionPointAfter(LS.Op);
  llvm::StringRef MangledName = LS.Op->getAttrOfType<mlir::StringAttr>(
                                  mlir::SymbolTable::getSymbolAttrName())
                                  .getValue();
  Builder.create<PushContOp>(Loc, MangledName, LS.Captures);
  LambdaScopes.pop_back();
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
    return SetError(Loc, "sequence must contain an expression", Body);
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

// walkDefineInits
//  - The BindingOps for the local defines have
//    been inserted by the `define` syntax. They are
//    initialized to "undefined" and their corresponding
//    heavy::Bindings placed in the EnvStack.
//    Walk the EnvStack up to the nearest EnvFrame and
//    collect these and insert the lazy initializers via SetOp
//    in the lexical order that they were defined.
void OpGen::walkDefineInits(Value Env) {
  Pair* P = cast<Pair>(Env);
  if (isa<EnvFrame>(P->Car)) return;
  walkDefineInits(P->Cdr);
  Binding* B = cast<Binding>(P->Car);
  mlir::Value BVal = BindingTable.lookup(B);
  mlir::Value Init = VisitDefineArgs(B->getValue());
  SourceLocation Loc = B->getValue().getSourceLocation();
  assert(BVal && "BindingTable should have an entry for local define");
  create<SetOp>(Loc, BVal, Init);
}

mlir::Value OpGen::createBody(SourceLocation Loc, Value Body) {
  IsTopLevel = false;
  {
    TailPosScope TPS(*this);
    IsTailPos = false;
    // Handle local defines.
    while (Pair* P = dyn_cast<Pair>(Body)) {
      Body = P;
      Symbol* S = dyn_cast_or_null<Symbol>(P->Car.car());
      Value LookupResult = S ? Context.Lookup(S).Value : Value();
      if (LookupResult != HEAVY_BASE_VAR(define)) break;
      heavy::base::define(*this, cast<Pair>(P->Car));
      Body = P->Cdr;
    }

    walkDefineInits(Context.EnvStack);
  }

  return createSequence(Loc, Body);
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  SourceLocation SymbolLoc = B->getName()->getSourceLocation();
  mlir::Value BVal = create<BindingOp>(SymbolLoc, Init);
  BindingTable.insert(B, BVal);

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

  // If EnvStack isn't an Environment then there is local
  // scope information on top of it
  Environment* Env = cast<Environment>(Context.EnvStack);

#if 0 // FIXME Values should report mutability properly
  if (Env->isImmutable()) {
    return SetError("define used in immutable environment", OrigCall);
  }
#endif

  EnvEntry Entry = Env->Lookup(S);
  if (Entry.Value && Entry.MangledName) {
    return SetError("define overwrites immutable location", S);
  }

  // If the name already exists in the current module
  // then it behaves like `set!`
  if (Binding* B = dyn_cast_or_null<Binding>(Entry.Value)) {
    TailPosScope TPS(*this);
    IsTailPos = false;
    mlir::Value BVal = BindingTable.lookup(B);
    assert(BVal && "expecting BindingOp for Binding");
    mlir::Value Init = VisitDefineArgs(DefineArgs);
    return create<SetOp>(DefineLoc, BVal, Init);
  }

  heavy::Mangler Mangler(Context);
  std::string MangledName = Mangler.mangleVariable(getModulePrefix(), S);
  auto GlobalOp = create<heavy::GlobalOp>(TopLevelBuilder, DefineLoc,
                                          MangledName);
  mlir::Block& Block = *GlobalOp.addEntryBlock();

  {
    // set insertion point. add initializer
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&Block);

    Binding* B = Context.CreateBinding(S, DefineArgs);
    BindingTable.insert(B, GlobalOp);
    Env->Insert(B);
    if (mlir::Value Init = VisitDefineArgs(DefineArgs)) {
      create<ContOp>(DefineLoc, Init);
    }
  }

  return GlobalOp;
}

mlir::Value OpGen::createIf(SourceLocation Loc, Value Cond, Value Then,
                            Value Else) {
  // Cond
  mlir::Value CondResult;
  {
    TailPosScope TPS(*this);
    IsTailPos = false;
    CondResult = GetSingleResult(Cond);
  }

  bool TailPos = isTailPos();
  IsTopLevel = false;

  mlir::Block* ThenBlock = new mlir::Block();
  mlir::Block* ElseBlock = new mlir::Block();
  bool RequiresContinuation = false;

  {
    // Treat then/else regions as tail position since if there is
    // any ApplyOp contained therein the whole operation becomes
    // an IfContOp which is a terminator with its own initCont region.
    TailPosScope TPS(*this);
    IsTailPos = true;

    // Then
    {
      mlir::OpBuilder::InsertionGuard IG(Builder);
      Builder.setInsertionPointToStart(ThenBlock);
      mlir::Value Result = Visit(Then);
      if (isa<ApplyOp>(ThenBlock->back())) {
        RequiresContinuation = true;
      } else {
        create<ContOp>(Loc, Result);
      }
    }

    // Else
    {
      mlir::OpBuilder::InsertionGuard IG(Builder);
      Builder.setInsertionPointToStart(ElseBlock);
      mlir::Value Result = Visit(Else);
      if (isa<ApplyOp>(ElseBlock->back())) {
        RequiresContinuation = true;
      } else {
        create<ContOp>(Loc, Result);
      }
    }
  }

  if (!RequiresContinuation) {
    auto IfOp = create<heavy::IfOp>(Loc, CondResult);
    IfOp.thenRegion().push_back(ThenBlock);
    IfOp.elseRegion().push_back(ElseBlock);
    return IfOp;
  }

  auto IfContOp = create<heavy::IfContOp>(Loc, CondResult);
  IfContOp.thenRegion().push_back(ThenBlock);
  IfContOp.elseRegion().push_back(ElseBlock);

  if (TailPos) return mlir::Value();
  return createContinuation(IfContOp.initCont());
}

// LHS can be a symbol or a binding
mlir::Value OpGen::createSet(SourceLocation Loc, Value LHS,
                                                 Value RHS) {
  TailPosScope TPS(*this);
  IsTailPos = false;
  assert((isa<Binding>(LHS) || isa<Symbol>(LHS)) &&
      "expects a Symbol or Binding for LHS");
  // ExprVal must be evaluated first so
  // the Binding will be in the continuation scope
  mlir::Value ExprVal = GetSingleResult(RHS);
  mlir::Value BVal = GetSingleResult(LHS);
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
  EnvEntry Entry = Context.Lookup(S);
  SourceLocation Loc = S->getSourceLocation();

  if (!Entry) {
    String* Msg = Context.CreateString("unbound symbol '",
                                       S->getVal(), "'");
    return SetError(Msg, S);
  }

  if (Entry.MangledName) {
    {
      llvm::StringRef SymName = Entry.MangledName->getView();
      mlir::ModuleOp M = Context.OpGen->getTopLevel();
      Operation* G = M.lookupSymbol(SymName);
      if (!G) {
        // Lazily insert extern GlobalOps
        // at the beginning of the module.
        // Note that OpEval will never visit these.
        mlir::OpBuilder::InsertionGuard IG(Builder);
        Builder.setInsertionPointToStart(M.getBody());
        G = create<GlobalOp>(Loc, SymName).getOperation();
      }
      return LocalizeValue(Entry.Value, G->getResult(0));
    }
  }

  return GetSingleResult(Entry.Value);
}

mlir::Value OpGen::VisitBinding(Binding* B) {
  mlir::Value V = BindingTable.lookup(B);
  // V should be a value for a BindingOp or nothing
  // BindingOps are created in the `define` syntax

  assert(V && "binding must exist in BindingTable");
  return LocalizeValue(B, V);
}

mlir::Value OpGen::HandleCall(Pair* P) {
  heavy::SourceLocation Loc = P->getSourceLocation();
  bool TailPos = isTailPos();
  IsTopLevel = false;
  mlir::Value Fn = GetSingleResult(P->Car);
  llvm::SmallVector<mlir::Value, 16> Args;
  HandleCallArgs(P->Cdr, Args);
  ApplyOp Op = create<ApplyOp>(Loc, Fn, Args);
  if (TailPos) return mlir::Value();

  // create the continuation
  return createContinuation(Op.initCont());
}

mlir::Value OpGen::createContinuation(mlir::Region& initCont) {
  //
  // TODO The current context should be able to tell us the arity
  //      of the continuation defaulting to 1 (plus the closure arg)
  //
  // TODO Detect if the continuation should simply discard effects,
  //      accepting any arity
  mlir::FunctionType FT = createFunctionType(/*Arity=*/1,
                                             /*HasRestParam=*/false);
  std::string MangledName = mangleFunctionName(llvm::StringRef());

  Builder.createBlock(&initCont);

  // create the continuation's function
  // subsequent operations will be nested within
  // relying on previous insertion guards to pull us out
  auto F = create<mlir::FuncOp>(heavy::SourceLocation(), MangledName, FT);
  PushContinuationScope(F);
  mlir::Block* FuncEntry = F.addEntryBlock();
  Builder.setInsertionPointToStart(FuncEntry);
  // Results drops the Closure arg at the front
  return FuncEntry->getArguments()[1];
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
    Args.push_back(GetSingleResult(P->Car));
  }
  HandleCallArgs(P->Cdr, Args);
}

mlir::Value OpGen::VisitPair(Pair* P) {
  Value Operator = P->Car;
  // A named operator might point to some kind of syntax transformer.
  if (Symbol* Name = dyn_cast<Symbol>(Operator)) {
    EnvEntry Entry = Context.Lookup(Name);
    if (!Entry) {
      // this makes it fail before the operands do
      String* Msg = Context.CreateString("unbound symbol '",
                                         Name->getVal(), "'");
      return SetError(Msg, Name);
    } else if (Binding* B = dyn_cast<Binding>(Entry.Value)) {
      Operator = B->getValue();
    } else {
      Operator = Entry.Value;
    }
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
mlir::Value OpGen::LocalizeValue(heavy::Value B, mlir::Value V) {
  mlir::Operation* Op = V.getDefiningOp();
  assert(Op && "value should be an operation result");

  mlir::Operation* Owner = Op
    ->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();

  return LocalizeRec(B, Op, Owner, LambdaScopes.rbegin());
}

mlir::Value OpGen::LocalizeRec(heavy::Value B,
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
  } else if (auto LG = dyn_cast<LoadGlobalOp>(Op)) {
    // Just make a new load global with the same name.
    NewVal = create<LoadGlobalOp>(Loc, LG.name()); 
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

heavy::Value heavy::eval(Context& C, Value V, Value EnvStack) {
  heavy::Value Args[2] = {V, EnvStack};
  int ArgCount = EnvStack ? 2 : 1;
  base::eval(C, ValueRefs(Args, ArgCount));
  return C.getCurrentResult();
}
