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

#include "PatternTemplate.h"
#include "heavy/Builtins.h"
#include "heavy/Context.h"
#include "heavy/Dialect.h"
#include "heavy/Mangle.h"
#include "heavy/OpGen.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;

namespace {
void PushLibraryDecls(Context& C, OpGen& O, Value LibraryDecls) {
  
}
} // namespace

OpGen::OpGen(heavy::Context& C)
  : Context(C),
    ImportsBuilder(&(C.MlirContext)),
    ModuleBuilder(&(C.MlirContext)),
    Builder(&(C.MlirContext)),
    BindingTable()
{
  C.MlirContext.loadDialect<heavy::Dialect>();
  mlir::Location Loc = Builder.getUnknownLoc();
  mlir::ModuleOp TopModule = Builder.create<mlir::ModuleOp>(Loc);
  Builder.setInsertionPointToStart(TopModule.getBody());
  mlir::ModuleOp M = Builder.create<mlir::ModuleOp>(Loc);
  ModuleOp = M;
  ModuleBuilder.setInsertionPointToStart(M.getBody());
  ImportsBuilder.setInsertionPointToStart(TopModule.getBody());
  Builder.clearInsertionPoint();
  LambdaScopes.emplace_back(ModuleOp, BindingTable);
}

std::string OpGen::mangleModule(heavy::Value NameSpec) {
  heavy::Mangler Mangler(Context);
  return Mangler.mangleModule(NameSpec);
}

std::string OpGen::mangleFunctionName(llvm::StringRef Name) {
  heavy::Mangler Mangler(Context);
  if (Name.empty()) {
    return Mangler.mangleAnonymousId(getModulePrefix(), LambdaNameCount++);
  }
  return Mangler.mangleFunction(getModulePrefix(), Name);
}

mlir::ModuleOp OpGen::getModuleOp() {
  return cast<mlir::ModuleOp>(ModuleOp);
}

mlir::Value OpGen::GetSingleResult(heavy::Value V) {
  TailPosScope TPS(*this);
  IsTailPos = false;
  mlir::Value Result = Visit(V);
  // mlir::Value() is only returned for expressions in tail position
  //               or syntax that should not be used outside proper
  //               context such as `import`, `define-syntax`, etc.
  if (!Result) {
      return SetError("expecting expression", V);
  }
  if (auto BlockArg = Result.dyn_cast<mlir::BlockArgument>()) {
    // the size includes the closure object
    if (BlockArg.getOwner()->getArguments().size() != 2) {
      return SetError("invalid continuation arity", V);
    }
  }
  return Result;
}

mlir::ValueRange OpGen::ExpandResults(mlir::Value Result) {
  if (Result.isa<mlir::OpResult>()) {
    return Result.getDefiningOp()->getResults();
  }
  else {
    auto BlockArg = Result.cast<mlir::BlockArgument>();
    return BlockArg.getOwner()->getArguments().drop_front();
  }
}

mlir::Value OpGen::GetPatternVar(heavy::Symbol* S) {
  heavy::Value SC = Context.Lookup(S).Value;
  if (Binding* B = dyn_cast<Binding>(SC)) {
    SC = B->getValue();
  }
  if (auto* Op = dyn_cast<mlir::Operation>(SC)) {
    return cast<SyntaxClosureOp>(Op);
  }
  llvm_unreachable("syntax closure not found in binding table");
}

void OpGen::VisitLibrary(heavy::SourceLocation Loc, std::string MangledName,
                         Value LibraryDecls) {
  assert(ModulePrefix.size() == 0 && "nested libraries are not allowed");
  assert(isTopLevel() && "library should be top level");
  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder.getContext());

  auto NewOpGen = std::make_unique<heavy::OpGen>(Context);
  auto NewEnv = std::make_unique<heavy::Environment>();
  NewOpGen->setModulePrefix(std::move(MangledName));

  // Create the output ModuleOp
  auto ModuleOp = ImportsBuilder.create<mlir::ModuleOp>(MLoc);
  ModuleBuilder.setInsertionPointToStart(ModuleOp.getBody());

  Context.WithEnv(std::make_unique<heavy::Environment>(),
    Context.CreateLambda([](heavy::Context& C, ValueArgs) {
      Value LibraryDecls = C.getCapture(0);
      C.WithOpGen(C.CreateLambda([](heavy::Context& C, ValueArgs) {
        Value LibraryDecls = C.getCapture(0);
        // The new OpGen needs a starting point for top level directives
        // for the given module/library/thing???

        // Recursively visit LibraryDecls
      }, CaptureList{LibraryDecls}));
      C.Cont();
    }, CaptureList{LibraryDecls}));
  // FIXME This does not currently support non-blocking directives.
  mlir::OpBuilder::InsertionGuard IG(ModuleBuilder);
  while (Pair* P = dyn_cast<Pair>(LibraryDecls)) {
    Visit(P->Car);
    LibraryDecls = P->Cdr;
  }
  if (!isa<Empty>(LibraryDecls)) {
    SetError(Loc,
             "expected proper list for library declarations",
             LibraryDecls);
  }
}

void OpGen::VisitTopLevel(Value V) {
  IsTopLevelAllowed = true;
  TopLevelOp = nullptr;

  // We use a null Builder to denote that we should
  // insert into a lazily created CommandOp by default
  // The insertion point is saved manually (ie not RAII)
  // and then restored in the continuation.
  mlir::OpBuilder::InsertPoint PrevInsertPoint = Builder.saveInsertionPoint();
  Builder.clearInsertionPoint();

  Context.PushCont([this, PrevInsertPoint](heavy::Context& Ctx, ValueRefs) {
    // Instead of the continuation argument we use TopLevelOp
    FinishTopLevelOp();
    if (TopLevelOp && TopLevelHandler) {
      // Call the TopLevelHandler
      Value Result = TopLevelOp;
      Ctx.Apply(TopLevelHandler, Result);
    } else {
      Ctx.Cont();
    }

    Builder.restoreInsertionPoint(PrevInsertPoint);
  });

  // Top Level Syntax may be async and use continuations.
  // (e.g. for `import` etc.)
  if (Pair* P = dyn_cast<Pair>(V)) {
    Value Operator = P->Car;
    if (isa<Syntax>(Operator)) {
      Value Input = P;
      Context.Apply(Operator, Input);
      return;
    }
  }

  // At this point we stipulate any custom syntax
  // routines are strictly sync/blocking.
  // The pushed continuation will handle the result.
  Visit(V);
  Context.Cont();
}

void OpGen::FinishTopLevelOp() {
  assert((TopLevelOp == nullptr || isa<CommandOp, GlobalOp>(TopLevelOp)) &&
      "Top level operation must be CommandOp or GlobalOp");

  if (heavy::CommandOp CommandOp =
        dyn_cast_or_null<heavy::CommandOp>(TopLevelOp)) {
    // Ensure current function body (if any) has a terminator.
    if (auto F = dyn_cast_or_null<FuncOp>(LambdaScopes.back().Op)) {
      mlir::Block& Block = F.getBody().back();
      if (!Block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::OpBuilder::InsertionGuard IG(Builder);
        Builder.setInsertionPointToEnd(&Block);
        create<ContOp>(heavy::SourceLocation(), createUndefined());
      }
    }

    // Ensure the CommandOp body has a terminator.
    mlir::Block& Block = CommandOp.body().front();
    assert(!Block.empty() && "command op must have body");
    if (!Block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder::InsertionGuard IG(Builder);
      Builder.setInsertionPointToEnd(&Block);
      create<ContOp>(heavy::SourceLocation(), createUndefined());
    }
  }

  // Pop continuation scopes to the TopLevelOp.
  while (LambdaScopes.size() > 0) {
    if (LambdaScopes.back().Op == TopLevelOp) {
      // Pop the TopLevelOp.
      LambdaScopes.pop_back();
      break;
    } else if (LambdaScopes.back().Op == ModuleOp) {
      break;
    } else {
      PopContinuationScope();
    }
  }
}

void OpGen::VisitTopLevelSequence(Value List) {
  if (Pair* P = dyn_cast<Pair>(List)) {
    Context.PushCont([](heavy::Context& C, ValueRefs) {
      Value Rest = C.getCapture(0);
      C.OpGen->VisitTopLevelSequence(Rest);
    }, CaptureList{P->Cdr});
    VisitTopLevel(P->Car);
  }
  else if (isa<Empty>(List)) {
    Context.Cont();
  } else {
    SetError("expected proper list in sequence", List);
  }
}

void OpGen::InsertTopLevelCommandOp(SourceLocation Loc) {
  auto CommandOp = createTopLevel<heavy::CommandOp>(Loc);
  setTopLevelOp(CommandOp.getOperation());
  mlir::Block& Block = *CommandOp.addEntryBlock();
  // overwrites Builder without reverting it
  Builder.setInsertionPointToStart(&Block);
}

mlir::Value OpGen::createUndefined() {
  return createHelper<UndefinedOp>(Builder, SourceLocation());
}

mlir::FunctionType OpGen::createFunctionType(unsigned Arity,
                                             bool HasRestParam) {
  mlir::Type ClosureT   = Builder.getType<HeavyValueTy>();
  mlir::Type ValueT     = Builder.getType<HeavyValueTy>();
  mlir::Type RestT      = Builder.getType<HeavyValueTy>();

  llvm::SmallVector<mlir::Type, 16> Types{};
  // push the closure type
  Types.push_back(ClosureT);
  if (Arity > 0) {
    for (unsigned i = 0; i < Arity - 1; i++) {
      Types.push_back(ValueT);
    }
    mlir::Type LastParamT = HasRestParam ? ValueT : RestT;
    Types.push_back(LastParamT);
  }

  return Builder.getFunctionType(Types, ValueT);
}

mlir::Value OpGen::createLambda(Value Formals, Value Body,
                                SourceLocation Loc,
                                llvm::StringRef Name) {
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
      if (IsLocalDefineAllowed) {
        FinishLocalDefines();
      }
      Result = LocalizeValue(Result);
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
  assert(isa<FuncOp>(LS.Op) && "expecting a function scope");
  Builder.setInsertionPointAfter(LS.Op);
  llvm::StringRef MangledName = LS.Op->getAttrOfType<mlir::StringAttr>(
                                  mlir::SymbolTable::getSymbolAttrName())
                                  .getValue();
  Builder.create<PushContOp>(Loc, MangledName, LS.Captures);
  LambdaScopes.pop_back();
}

bool OpGen::isLocalDefineAllowed() {
  return IsLocalDefineAllowed;
#if 0
  mlir::Block* Block = Builder.getInsertionBlock();
  if (!Block) return false;
  return (Block->empty() ||
          isa<BindingOp>(Block->back()));
#endif
}

// createSyntaxSpec
//  - SyntaxSpec is the full <Keyword> <TransformerSpec> pair
//    that is passed to the syntax function.
//  - OrigCall is likely (define-syntax ...) or (let-syntax ...)
mlir::Value OpGen::createSyntaxSpec(Pair* SyntaxSpec, Value OrigCall) {
  mlir::Value Result;
  Environment* TopLevelEnv = nullptr;
  if (isTopLevel()) {
    TopLevelEnv = cast<Environment>(Context.EnvStack);
  }

  Symbol* Keyword = dyn_cast<Symbol>(SyntaxSpec->Car);
  if (!Keyword) return SetError("expecting syntax spec keyword", SyntaxSpec);
  Pair* TransformerSpec = dyn_cast<Pair>(SyntaxSpec->Cdr.car());
  if (!TransformerSpec || !isa<Empty>(SyntaxSpec->Cdr.cdr())) {
    return SetError("invalid syntax spec", SyntaxSpec);
  }
  if (Symbol* TS = dyn_cast_or_null<Symbol>(TransformerSpec->Car)) {
    // Operator is typically syntax-rules here.
    if (Value Operator = Context.Lookup(TS).Value) {
      // Invoke the syntax with the entire SyntaxSpec
      // so syntax-rules et al. can know the Keyword.
      switch (Operator.getKind()) {
      case ValueKind::BuiltinSyntax: {
        BuiltinSyntax* BS = cast<BuiltinSyntax>(Operator);
        Result = BS->Fn(*this, SyntaxSpec);
        break;
      }
      case ValueKind::Syntax: {
        Value Input = SyntaxSpec;
        Context.Run(Operator, ValueRefs(Input));
        Result = toValue(Context.getCurrentResult());
        break;
      }
      default: {
        Result = mlir::Value();
        break;
      }}
    }
  }

  if (Context.CheckError()) return Result;
  if (!Result) {
    return SetError("expecting syntax object", SyntaxSpec);
  }

  auto SyntaxOp = Result.getDefiningOp<heavy::SyntaxOp>();
  if (!SyntaxOp) {
    return SetError("expecting syntax object", SyntaxSpec);
  }

  auto Fn = [SyntaxOp](heavy::Context& C, ValueRefs Args) -> void {
    heavy::Value Input = Args[0];
    invokeSyntaxOp(C, SyntaxOp, Input);
    // The contained OpGenOp will call C.Apply(...).
  };
  heavy::Syntax* Syntax = Context.CreateSyntax(Fn);

  if (TopLevelEnv) {
    TopLevelEnv->SetSyntax(Keyword, Syntax);
  } else {
    Binding* B = Context.CreateBinding(Keyword, Syntax);
    Context.PushLocalBinding(B);
  }

  return mlir::Value();
}

mlir::Value OpGen::createSyntaxRules(SourceLocation Loc,
                                     Symbol* Keyword,
                                     Symbol* Ellipsis, 
                                     heavy::Value LiteralList,
                                     heavy::Value PatternDefs) {
  // Expect list of unique literal identifiers.
  llvm::SmallPtrSet<String*, 4> Literals;
  auto insertLiteral = [&](Value V) -> bool {
    Symbol* S = dyn_cast<Symbol>(V);
    if (!S) {
      SetError("expecting keyword literal", V);
      return true;
    }
    if (S->equals(Ellipsis)) {
      SetError("keyword literal cannot be same as ellipsis", S);
      return true;
    }
    bool Inserted;
    std::tie(std::ignore, Inserted) = Literals.insert(S->getString());
    if (!Inserted) {
      SetError("keyword specified multiple times", S);
      return true;
    }
    return false;
  };
  while (Pair* X = dyn_cast<Pair>(LiteralList)) {
    Context.setLoc(X);
    if (insertLiteral(X->Car)) {
      return Error();
    }
    LiteralList = X->Cdr;
  }
  if (!isa<Empty>(LiteralList)) {
    return SetError("expecting keyword list", LiteralList);
  }

  // Create the SyntaxOp
  auto SyntaxOp = create<heavy::SyntaxOp>(Loc);
  mlir::Block& Block = SyntaxOp.region().emplaceBlock();

  // TODO create anonymous function name for SyntaxOp symbol
  mlir::BlockArgument Arg = Block.addArgument(
                                    Builder.getType<HeavyValueTy>());
  mlir::OpBuilder::InsertionGuard IG(Builder);
  Builder.setInsertionPointToStart(&Block);

  // Iterate through each pattern/template pair.
  assert(!Context.CheckError() && "should not have errors here");
  while (Pair* I = dyn_cast<Pair>(PatternDefs)) {
    heavy::Pair* X        = dyn_cast<Pair>(I->Car);
    heavy::Value Pattern  = X->Car;
    heavy::Value Template = X->Cdr.car();
    bool IsProperPT = isa_and_nonnull<Empty>(X->Cdr.cdr());
    if (!IsProperPT) {
      return SetError("expecting pattern template pair", X);
    }

    auto PatternOp = create<heavy::PatternOp>(Pattern.getSourceLocation());
    mlir::Block& B = PatternOp.region().emplaceBlock();
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&B);
    PatternTemplate PT(*this, Keyword, Ellipsis, Literals);

    PT.VisitPatternTemplate(Pattern, Template, Arg);
    PatternDefs = I->Cdr;
  }
  if (!isa<Empty>(PatternDefs) || Block.empty()) {
    return SetError("expecting list of pattern templates pairs", PatternDefs);
  }
  return SyntaxOp.result();
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

// WalkDefineInits
//  - The BindingOps for the local defines have
//    been inserted by the `define` syntax. They are
//    initialized to "undefined" and their corresponding
//    heavy::Bindings placed in the EnvStack.
//    Walk the EnvStack up to the nearest EnvFrame and
//    collect these and insert the lazy initializers via SetOp
//    in the lexical order that they were defined.
bool OpGen::WalkDefineInits(Value Env,
                            llvm::SmallPtrSetImpl<String*>& LocalNames) {
  Pair* P = cast<Pair>(Env);
  if (isa<EnvFrame>(P->Car)) return false;
  if (WalkDefineInits(P->Cdr, LocalNames)) return true;

  // Check for duplicate names.
  Binding* B = cast<Binding>(P->Car);
  Symbol* Name = B->getName();
  String* Identifier = Name->getString();
  bool IsNameInserted;
  std::tie(std::ignore, IsNameInserted) = LocalNames.insert(Identifier);
  if (!IsNameInserted) {
    SetError("local variable has duplicate definitions", Name);
    return true;
  }

  // Insert the binding initializer.
  mlir::Value Init = VisitDefineArgs(B->getValue());
  mlir::Value BVal = LocalizeValue(BindingTable.lookup(B), B);
  SourceLocation Loc = B->getValue().getSourceLocation();
  assert(BVal && "BindingTable should have an entry for local define");
  create<SetOp>(Loc, BVal, Init);
  return false;
}

bool OpGen::FinishLocalDefines() {
  TailPosScope TPS(*this);
  IsTailPos = false;
  IsLocalDefineAllowed = false;
  llvm::SmallPtrSet<String*, 8> LocalNames;
  return WalkDefineInits(Context.EnvStack, LocalNames);
}

mlir::Value OpGen::createBody(SourceLocation Loc, Value Body) {
  IsLocalDefineAllowed = true;
  return createSequence(Loc, Body);
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  SourceLocation SymbolLoc = B->getName()->getSourceLocation();
  // Bypass the simpler create<Op> overload to avoid triggering
  // the finalization of the local defines.
  mlir::Value BVal = createHelper<BindingOp>(Builder, SymbolLoc, Init);
  BindingTable.insert(B, BVal);

  return BVal;
}

mlir::Value OpGen::createDefine(Symbol* S, Value DefineArgs,
                                           Value OrigCall) {
  if (isTopLevel()) return createTopLevelDefine(S, DefineArgs, OrigCall);
  if (!IsLocalDefineAllowed) return SetError("unexpected define", OrigCall);
  // Create the binding with a lazy init.
  // (Include everything after the define
  //  keyword to visit it later because it could
  //  be a terse lambda syntax.)
  Binding* B = Context.CreateBinding(S, DefineArgs);
  // Push to the local environment.
  Context.PushLocalBinding(B);
  mlir::Value BVal = createBinding(B, mlir::Value());
  assert(IsLocalDefineAllowed && "define should still be allowed");
  return BVal;
}

mlir::Value OpGen::createTopLevelDefine(Symbol* S, Value DefineArgs,
                                        Value OrigCall) {
  SourceLocation DefineLoc = OrigCall.getSourceLocation();
  if (!IsTopLevelAllowed) {
    return SetError("unexpected define", OrigCall);
  }

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
    mlir::Value Init = VisitDefineArgs(DefineArgs);
    mlir::Value BVal = LocalizeValue(BindingTable.lookup(B), B);
    assert(BVal && "expecting BindingOp for Binding");
    return create<SetOp>(DefineLoc, BVal, Init);
  }

  heavy::Mangler Mangler(Context);
  std::string MangledName = Mangler.mangleVariable(getModulePrefix(), S);
  auto GlobalOp = createTopLevel<heavy::GlobalOp>(DefineLoc, MangledName);
  setTopLevelOp(GlobalOp.getOperation());
  mlir::Block& Block = *GlobalOp.addEntryBlock();

  {
    // set insertion point. add initializer
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&Block);

    Binding* B = Context.CreateBinding(S, DefineArgs);
    // Insert into the module level scope
    BindingTable.insertIntoScope(&LambdaScopes[0].BindingScope_, B,
                                 GlobalOp);
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
      if (!ThenBlock->empty() && isa<ApplyOp>(ThenBlock->back())) {
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
      if (!ElseBlock->empty() && isa<ApplyOp>(ElseBlock->back())) {
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

// createGlobal - Create or load a global idempotently.
//                Prevent duplicate Ops in an isolated scope.
mlir::Value OpGen::createGlobal(SourceLocation Loc,
                                llvm::StringRef MangledName) {
  mlir::ModuleOp M = getModuleOp();
  Operation* G = LookupSymbol(MangledName);

  if (!G) {
    // Lazily insert extern GlobalOps
    // at the beginning of the module.
    // Note that OpEval will never visit these.
    mlir::OpBuilder::InsertionGuard IG(ModuleBuilder);
    ModuleBuilder.setInsertionPointToStart(M.getBody());
    G = createTopLevel<GlobalOp>(Loc, MangledName)
      .getOperation();
  }

  assert(isa<GlobalOp>(G) && "symbol should point to global");

  // Localize globals by the GlobalOp's Operation*
  auto CanonicalValue = heavy::Value(G);
  mlir::Value LocalV = BindingTable.lookup(CanonicalValue);
  mlir::Value V = LocalV ? LocalV : G->getResult(0);
  return LocalizeValue(V, CanonicalValue);
}

mlir::Value OpGen::VisitExternName(ExternName* EN) {
  return createGlobal(EN->getSourceLocation(), EN->getView());
}

mlir::Value OpGen::VisitSyntaxClosure(SyntaxClosure* SC) {
  Value PrevEnv = Context.getEnvironment();
  Context.setEnvironment(SC->Env);
  auto ScopeExit = llvm::make_scope_exit([&] {
    Context.setEnvironment(PrevEnv);
  });
  return Visit(SC->Node);
}

mlir::Value OpGen::VisitSymbol(Symbol* S) {
  EnvEntry Entry = Context.Lookup(S);
  SourceLocation Loc = S->getSourceLocation();

  if (!Entry) {
    // Default to the name of a global
    heavy::Mangler Mangler(Context);
    std::string MangledName = Mangler.mangleVariable(getModulePrefix(), S);
    return createGlobal(Loc, MangledName);
  }
  return VisitEnvEntry(Loc, Entry);
}

mlir::Value OpGen::VisitEnvEntry(heavy::SourceLocation Loc, EnvEntry Entry) {
  if (Entry.MangledName) {
    llvm::StringRef SymName = Entry.MangledName->getView();
    return createGlobal(Loc, SymName);
  }

  return GetSingleResult(Entry.Value);
}

mlir::Value OpGen::VisitBinding(Binding* B) {
  mlir::Value V = BindingTable.lookup(B);
  // V should be a value for a BindingOp or nothing
  // BindingOps are created in the `define` syntax

  assert(V && "binding must exist in BindingTable");
  return LocalizeValue(V, B);
}

mlir::Value OpGen::HandleCall(Pair* P) {
  heavy::SourceLocation Loc = P->getSourceLocation();
  ApplyOp Op;

  {
    TailPosScope TPS(*this);
    IsTailPos = false;

    mlir::Value Fn = GetSingleResult(P->Car);
    llvm::SmallVector<mlir::Value, 16> Args;

    Value V = P->Cdr;
    while (auto* P2 = dyn_cast<Pair>(V)) {
      mlir::Value Arg = GetSingleResult(P2->Car);
      Args.push_back(Arg);
      V = P2->Cdr;
    }
    if (!isa<Empty>(V)) {
      return SetError("improper list as call expression", V);
    }

    // Localize all of the operands
    Fn = LocalizeValue(Fn);
    for (mlir::Value& Arg : Args) {
      Arg = LocalizeValue(Arg);
    }
    Op = create<ApplyOp>(Loc, Fn, Args);
  }

  if (IsTailPos) return mlir::Value();
  return createContinuation(Op.initCont());
}

mlir::Value OpGen::createOpGen(SourceLocation Loc, mlir::Value Input) {
  // OpGenOp should always be considered tail position.
  create<heavy::OpGenOp>(Loc, Input);
  return mlir::Value();
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

mlir::Value OpGen::VisitPair(Pair* P) {
  Value Operator = P->Car;
  // A named operator might point to some kind of syntax transformer.
  if (Symbol* Name = dyn_cast<Symbol>(Operator)) {
    EnvEntry Entry = Context.Lookup(Name);
    if (!Entry) {
      Operator = Undefined();
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
    case ValueKind::Syntax: {
      Value Input = P;
      Context.Run(Operator, ValueRefs(Input));
      return toValue(Context.getCurrentResult());
    }
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
mlir::Value OpGen::LocalizeValue(mlir::Value V, heavy::Value B) {
  mlir::Operation* Owner;
  LambdaScopeIterator LSI = LambdaScopes.rbegin();

  if (mlir::Operation* Op = V.getDefiningOp()) {
    heavy::SourceLocation Loc = {};
    LambdaScopeNode& LS = *LSI;
    Owner = Op->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
    if (LS.Op == Owner) return V;

    mlir::Value NewVal;
    if (auto G = dyn_cast<GlobalOp>(Op)) {
      NewVal = create<LoadGlobalOp>(Loc, G.getName());
    } else if (auto LG = dyn_cast<LoadGlobalOp>(Op)) {
      // Just make a new load global with the same name.
      NewVal = create<LoadGlobalOp>(Loc, LG.name()); 
    }
    if (NewVal) {
      if (B) {
        // Prevent duplicate LoadGlobalOps in the same scope
        // by shadowing them in the BindingTable.
        LambdaScopeNode& LS = *LSI;
        BindingTable.insertIntoScope(&LS.BindingScope_, B, NewVal);
      }
      return NewVal;
    }
  } else {
    mlir::BlockArgument BlockArg = V.cast<mlir::BlockArgument>();
    Owner = BlockArg.getOwner()->getParentOp();
  }

  // Non-globals must be captured in every closure up to the scope
  // where they were created.
  return LocalizeRec(B, V, Owner, LSI);
}

mlir::Value OpGen::LocalizeRec(heavy::Value B,
                               mlir::Value V,
                               mlir::Operation* Owner,
                               LambdaScopeIterator LSI) {
  assert(LSI != LambdaScopes.rend() && "value must be in a scope");
  LambdaScopeNode& LS = *LSI;
  if (LS.Op == Owner) return V;

  heavy::SourceLocation Loc = {};

  mlir::Value ParentLocal = LocalizeRec(B, V, Owner, ++LSI);

  // Capture V for use in the current function scope.
  mlir::OpBuilder::InsertionGuard IG(Builder);
  auto FuncOp = cast<mlir::FuncOp>(LS.Op);
  mlir::Block& Block = FuncOp.getBody().front();
  Builder.setInsertionPointToStart(&Block);
  LS.Captures.push_back(ParentLocal);
  uint32_t Index = LS.Captures.size() - 1;
  mlir::Value Closure = Block.getArguments().front();
  mlir::Value NewVal = create<LoadClosureOp>(Loc, Closure, Index);

  if (B) {
    BindingTable.insertIntoScope(&LS.BindingScope_, B, NewVal);
  }
  return NewVal;
}

mlir::Operation* OpGen::LookupSymbol(llvm::StringRef MangledName) {
  mlir::ModuleOp M = getModuleOp();
  Operation* G = M.lookupSymbol(MangledName);
  if (G) return G;

  M = cast<mlir::ModuleOp>(M->getParentOp());
  return M.lookupSymbol(MangledName);
}
