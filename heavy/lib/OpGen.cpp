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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace heavy;
// LambdaScope - RAII object that pushes an operation that is
//               FunctionLike to the scope stack along with a
//               BindingScope where we can insert stack local
//               values resulting from operations that load from
//               a global or variable captured by a closure.
//
//               Note that BindingScope is still used for
//               non-isolated scopes (e.g. `let` syntax).
struct OpGen::LambdaScope {
  OpGen& O;
  LambdaScopeNode& Node;

  LambdaScope(OpGen& O, mlir::Operation* Op)
    : O(O),
      Node((O.LambdaScopes.emplace_back(Op, /*CallOp*/nullptr,
                                        O.BindingTable),
            O.LambdaScopes.back()))
  { }

  ~LambdaScope() {
    // pop all intermediate continuation scopes
    // and then our own lambda scope
    while (O.LambdaScopes.size() > 0) {
      mlir::Operation* CurOp = O.LambdaScopes.back().Op;
      if (CurOp == Node.Op) {
        O.LambdaScopes.pop_back();
        return;
      } else {
        O.PopContinuationScope();
      }
    }
    llvm_unreachable("scope should be on stack");
  }
};

OpGen::OpGen(heavy::Context& C, heavy::Symbol* ModulePrefix)
  : Context(C),
    ModuleBuilder(C.MLIRContext.get()),
    Builder(C.MLIRContext.get()),
    BindingTable(),
    ModulePrefix(ModulePrefix)
{
  mlir::Location Loc = Builder.getUnknownLoc();
  mlir::OpBuilder ImportsBuilder(C.MLIRContext.get());
  mlir::ModuleOp TopModule;
  if (!C.OpGen) {
    assert(!C.ModuleOp && "There should be only one top level module");
    C.MLIRContext->loadDialect<heavy::Dialect, mlir::func::FuncDialect>();
    // Create the module that contains the main module and import modules
    TopModule = Builder.create<mlir::ModuleOp>(Loc);
    C.ModuleOp = TopModule;
    // The first child ModuleOp is the main module. All subsequent
    // modules are imports.
  } else {
    TopModule = cast<mlir::ModuleOp>(C.ModuleOp);
  }
  ImportsBuilder.setInsertionPointToEnd(TopModule.getBody());
  ModuleOp = ImportsBuilder.create<mlir::ModuleOp>(Loc,
                                        getModulePrefix());

  ModuleBuilder.setInsertionPointToStart(
      cast<mlir::ModuleOp>(ModuleOp).getBody());
  LambdaScopes.emplace_back(ModuleOp, /*CallOp=*/nullptr, BindingTable);
}

OpGen::~OpGen() {
  assert((LambdaScopes.size() == 1 || CheckError())
      && "LambdaScopes should be unwound");
  // Unwind if there was an error.
  while (LambdaScopes.size() > 1)
    LambdaScopes.pop_back();
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

std::string OpGen::mangleVariable(heavy::Value Name) {
  heavy::Mangler Mangler(Context);
  return Mangler.mangleVariable(getModulePrefix(), Name);
}

// Mangle the variable name of a syntax object.
std::string OpGen::mangleSyntax(heavy::Value Name) {
  heavy::Mangler Mangler(Context);
  return Mangler.mangleAnonymousId(getModulePrefix(), LambdaNameCount++);
}

mlir::ModuleOp OpGen::getModuleOp() {
  return cast<mlir::ModuleOp>(ModuleOp);
}

void OpGen::createLoadModule(heavy::SourceLocation Loc,
                             heavy::Symbol* MangledName) {
  TopLevelOp = createHelper<heavy::LoadModuleOp>(ModuleBuilder, Loc,
                                    MangledName->getStringRef());
}

mlir::Value OpGen::GetSingleResult(heavy::Value V) {
  TailPosScope TPS(*this);
  IsTailPos = false;
  mlir::Value Result = Visit(V);

  // Take the first continuation arg.
  if (Result && isa<HeavyValueRefsTy>(Result.getType()))
    Result = create<LoadRefOp>(heavy::SourceLocation(), Result, /*Index*/0);

  if (CheckError())
    return Error();
  // mlir::Value() is only returned for expressions in tail position
  //               or syntax that should not be used outside proper
  //               context such as `import`, `define-syntax`, etc.
  if (!Result) {
      return SetError("expecting expression", V);
  }

  return LocalizeValue(Result);
}

// WithLibraryEnv - Call a thunk within the library environment
//                  for <library spec>. (ie begin, import, export)
void OpGen::WithLibraryEnv(Value Thunk) {
  if (!LibraryEnvProc) {
    SetError("not in a library context");
    return;
  }
  assert(!isa<Undefined>(LibraryEnvProc->getValue()) &&
      "LibraryEnvProc must be initialized");
  Context.CallCC(Context.CreateLambda([](heavy::Context& C, ValueRefs Args) {
    Value CC = Args[0];
    Value Thunk = C.getCapture(0);
    Binding* LibraryEnvProc = cast<Binding>(C.getCapture(1));

    Value NewThunk = C.CreateLambda([](heavy::Context& C, ValueRefs) {
      Value Thunk = C.getCapture(0);
      Value CC = C.getCapture(1);
      C.PushCont(CC);  // Resume the current continuation.
      C.ApplyThunk(Thunk);
    }, CaptureList{Thunk, CC});

    C.Apply(LibraryEnvProc->getValue(), NewThunk);
  }, CaptureList{Thunk, LibraryEnvProc}));
}

void OpGen::VisitLibrary(heavy::SourceLocation Loc,
                         heavy::Symbol* MangledName,
                         Value LibraryDecls) {
  assert(isTopLevel() && "library should be top level");
  // LibraryEnvProc - Capture and later set to the escape procedure
  //                  that is used to process top level exprs for the
  //                  library which must be with a (begin ...) syntax.
  assert(!LibraryEnvProc && "should not already be in library context");
  LibraryEnvProc = Context.CreateBinding(Undefined());
  Value HandleLibraryDecls = Context.CreateBinding(Int(5));
  LibraryDecls = Context.CreateBinding(LibraryDecls);

  // PushCont the creation of LibraryEnvProc
  Context.PushCont([](heavy::Context& C, ValueRefs) {
    Binding* HandleLibraryDecls = cast<Binding>(C.getCapture(0));
    Value LibraryEnvProc = C.getCapture(1);
    Symbol* MangledName = cast<Symbol>(C.getCapture(2));

    if (isa<Empty>(HandleLibraryDecls->getValue())) {
      C.Cont();
      return;
    }

    Value Thunk = C.CreateLambda([](heavy::Context& C, ValueRefs) {
      Binding* HandleLibraryDecls = cast<Binding>(C.getCapture(0));
      Value LibraryEnvProc = C.getCapture(1);

      // PushCont the initial calling of HandleLibraryDecls
      C.PushCont(HandleLibraryDecls->getValue());

      C.SaveEscapeProc(LibraryEnvProc, [](heavy::Context& C, ValueRefs Args) {
        // Handle the sequence from `begin` from within the lib environment.
        Binding* HandleLibraryDecls = cast<Binding>(C.getCapture(0));
        if (isa<Lambda>(HandleLibraryDecls->getValue())) {
          C.PushCont(HandleLibraryDecls->getValue());
          C.PushCont([](heavy::Context& C, ValueRefs Args) {
            // Args[0] is Thunk from OpGen::WithLibraryEnv(Value Thunk)
            C.ApplyThunk(Args[0]);
          });
          C.Cont(Args[0]);
        } else {
          // If the escape proc is cleared then let this proc
          // complete so it deletes the environment.
          C.Cont();
        }
      }, CaptureList{HandleLibraryDecls});
    }, CaptureList{HandleLibraryDecls, LibraryEnvProc});
    // Taking ownership of EnvPtr
    auto EnvPtr = std::make_unique<heavy::Environment>(C, MangledName);
    C.WithEnv(std::move(EnvPtr), Thunk);
  }, CaptureList{HandleLibraryDecls, LibraryEnvProc, MangledName});


  // The library environment isn't used until we are in a sequence (begin).
  // Use the parent environment without allowing operations to be inserted.
  auto LibraryBefore = Context.CreateLambda([](heavy::Context& C, ValueRefs) {
    // Store LibraryEnvProc as a member so it is accessible within
    // library specs
    C.OpGen->LibraryEnvProc = cast<Binding>(C.getCapture(0));
    C.Cont();
  }, CaptureList{LibraryEnvProc});
  auto LibraryAfter = Context.CreateLambda([](heavy::Context& C, ValueRefs) {
    C.OpGen->LibraryEnvProc = nullptr;
    C.Cont();
  }, CaptureList{});
  auto LibraryThunk = Context.CreateLambda([](heavy::Context& C, ValueRefs) {
    Value HandleLibraryDecls = C.getCapture(0);
    Value LibraryDecls       = C.getCapture(1);
    C.SaveEscapeProc(HandleLibraryDecls, [](heavy::Context& C, ValueRefs) {
      Binding* HandleLibraryDecls = cast<Binding>(C.getCapture(0));
      Binding* LibraryDecls = cast<Binding>(C.getCapture(1));
      if (isa<Empty>(C.OpGen->LibraryEnvProc->getValue())) {
        C.Cont();
      } else if (Pair* P = dyn_cast<Pair>(LibraryDecls->getValue())) {
        LibraryDecls->setValue(P->Cdr);
        C.PushCont(HandleLibraryDecls->getValue());
        C.OpGen->VisitLibrarySpec(P->Car);
      } else if (isa<Empty>(LibraryDecls->getValue())) {
        // The library is done.
        // Unset stuff so the cleanups will run.
        cast<Binding>(HandleLibraryDecls)->setValue(Empty());
        C.OpGen->LibraryEnvProc->setValue(Empty());
        C.Cont();
      } else {
        C.OpGen->SetError("expected proper list for library declarations",
                          LibraryDecls->getValue());
      }
    }, CaptureList{HandleLibraryDecls, LibraryDecls});
  }, CaptureList{HandleLibraryDecls, LibraryDecls});

  Context.DynamicWind(LibraryBefore,
                      LibraryThunk,
                      LibraryAfter);
}

void OpGen::VisitLibrarySpec(Value LibSpec) {
  // Use PushCont to avoid destruction of this.
  Context.PushCont([](heavy::Context& C, ValueRefs) {
    Value LibSpec = C.getCapture(0);
    if (Pair* P = dyn_cast<Pair>(LibSpec)) {
      heavy::Value SyntaxOperator;
      if (isIdentifier(P->Car))
        SyntaxOperator = C.GetSyntax(C.OpGen->LookupEnv(P->Car)).Value;
      if (SyntaxOperator) {
        C.OpGen->CallSyntax(SyntaxOperator, P);
      } else {
        C.OpGen->SetError("expecting library spec", LibSpec);
        return;
      }
      if (C.OpGen->CheckError())
        return;
    }

    C.Cont();
  }, CaptureList{LibSpec});
  Context.Cont();
}

void OpGen::VisitTopLevel(Value V) {
  if (LibraryEnvProc) {
    VisitLibrarySpec(V);
    return;
  }
  TopLevelOp = nullptr;

  // We use a null Builder to denote that we should
  // insert into a lazily created CommandOp by default
  // The insertion point is saved manually (ie not RAII)
  // and then restored in the continuation.
  mlir::OpBuilder::InsertPoint PrevIp = Builder.saveInsertionPoint();
  Builder.clearInsertionPoint();

  Context.PushCont([this, PrevIp](heavy::Context& C, ValueRefs) {
    // Instead of the continuation argument we use TopLevelOp
    FinishTopLevelOp();
    if (TopLevelOp && TopLevelHandler) {
      // Call the TopLevelHandler
      Value Result = TopLevelOp;
      TopLevelOp = nullptr;
      C.Apply(TopLevelHandler, Result);
    } else {
      C.Cont();
    }

    Builder.restoreInsertionPoint(PrevIp);
  });
  Context.PushCont([](heavy::Context& C, ValueRefs) {
    heavy::Value V = C.getCapture(0);
    heavy::OpGen* OpGen = C.OpGen;  // For debug only.
    C.OpGen->Visit(V);
    assert(OpGen == C.OpGen && "OpGen should not unwind itself.");
    if (C.OpGen->CheckError())
      return;
    C.Cont();
  }, CaptureList{V});
  Context.Cont();
}

void OpGen::FinishTopLevelOp() {
  assert((TopLevelOp == nullptr ||
          isa<CommandOp, GlobalOp, LoadModuleOp>(TopLevelOp)) &&
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
    mlir::Block& Block = CommandOp.getBody().front();
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
  if (LibraryEnvProc) {
    WithLibraryEnv(Context.CreateLambda([](heavy::Context& C, ValueRefs) {
      C.OpGen->VisitTopLevelSequence(C.getCapture(0));
    }, CaptureList{List}));
    return;
  }
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
  if (Builder.getInsertionBlock() == nullptr)
    return mlir::Value();
  return createHelper<UndefinedOp>(Builder, SourceLocation());
}

mlir::FunctionType OpGen::createFunctionType(unsigned Arity,
                                             RestParamKind RPK) {
  mlir::Type ClosureT   = Builder.getType<HeavyContextTy>();
  mlir::Type ValueT     = Builder.getType<HeavyValueTy>();

  llvm::SmallVector<mlir::Type, 16> Types{};
  // push the closure type
  Types.push_back(ClosureT);
  if (Arity > 0) {
    for (unsigned i = 0; i < Arity - 1; i++) {
      Types.push_back(ValueT);
    }

    mlir::Type LastParamT;
    switch (RPK) {
    case RestParamKind::None:
      LastParamT = ValueT;
      break;
    case RestParamKind::List:
      LastParamT = Builder.getType<HeavyRestTy>();
      break;
    case RestParamKind::ValueRefs:
      LastParamT = Builder.getType<HeavyValueRefsTy>();
      break;
    }
    Types.push_back(LastParamT);
  }

  return Builder.getFunctionType(Types, ValueT);
}

heavy::FuncOp OpGen::createFunction(SourceLocation Loc,
                                    llvm::StringRef MangledName,
                                    mlir::FunctionType FT) {
  // Insert the FuncOp such that it precedes the current top level op.
  return createHelper<heavy::FuncOp>(ModuleBuilder, Loc, MangledName, FT);
}

mlir::Value OpGen::createLambda(Value Formals, Value Body,
                                SourceLocation Loc,
                                llvm::StringRef Name) {
  // Flush any internal definitions from containing body if any.
  if (IsLocalDefineAllowed)
    FinishLocalDefines();

  // Ensure we are no longer top level.
  if (!TopLevelOp)
    InsertTopLevelCommandOp(Loc);

  std::string MangledName = mangleFunctionName(Name);
  if (MangledName.empty())
    return Error();

  bool HasRestParam = false;
  heavy::EnvFrame* EnvFrame = Context.PushLambdaFormals(Formals, HasRestParam,
                                                        CurSyntaxClosure);
  if (!EnvFrame)
    return Error();
  unsigned Arity = EnvFrame->getBindings().size();
  RestParamKind RPK = HasRestParam ? RestParamKind::List
                                   : RestParamKind::None;
  mlir::FunctionType FT = createFunctionType(Arity, RPK);
  auto F = createFunction(Loc, MangledName, FT);
  llvm::SmallVector<mlir::Value, 8> Captures;

  // Insert into the function body
  {
    // Start new scope in tail pos
    TailPosScope TPS(*this);
    IsTailPos = true;
    LambdaScope LS(*this, F);
    mlir::OpBuilder::InsertionGuard IG(Builder);
    mlir::Block& Block = *F.addEntryBlock();
    Builder.setInsertionPointToStart(&Block);

    // ValueArgs drops the Closure arg at the front
    auto ValueArgs  = Block.getArguments().drop_front();
    // Create the BindingOps for the arguments
    for (auto tup : llvm::zip(EnvFrame->getBindings(),
                              ValueArgs)) {
      auto [B, Arg] = tup;
      createBinding(B, Arg);
    }

    // If Result is null then it already
    // has a terminator.
    if (mlir::Value Result = createBody(Loc, Body)) {
      // Flush the internal definitions if we have not done so.
      if (IsLocalDefineAllowed)
        FinishLocalDefines();
      Result = LocalizeValue(Result);
      Builder.create<ContOp>(Result.getLoc(), Result);
    }

    Context.PopEnvFrame();
    Captures = std::move(LS.Node.Captures);
  }

  return create<LambdaOp>(Loc, MangledName, Captures);
}

void OpGen::PopContinuationScope() {
  mlir::OpBuilder::InsertionGuard IG(Builder);
  LambdaScopeNode& LS = LambdaScopes.back();
  mlir::Location Loc = LS.Op->getLoc();
  assert(isa<FuncOp>(LS.Op) && "expecting a function scope");
  llvm::StringRef MangledName = LS.Op->getAttrOfType<mlir::StringAttr>(
                                  mlir::SymbolTable::getSymbolAttrName())
                                  .getValue();
  assert(LS.CallOp != nullptr && "a PushContOp should precede a call");
  Builder.setInsertionPoint(LS.CallOp);
  Builder.create<PushContOp>(Loc, MangledName, LS.Captures);
  LambdaScopes.pop_back();
}

bool OpGen::isLocalDefineAllowed() {
  return IsLocalDefineAllowed;
}

// Does not add the entry block (use addEntryBlock)
heavy::FuncOp OpGen::createSyntaxFunction(SourceLocation Loc) {
  std::string MangledName = mangleFunctionName("");
  if (MangledName.empty())
    return heavy::FuncOp();
  mlir::FunctionType FT = createFunctionType(/*Arity=*/2, RestParamKind::None);
  return createFunction(Loc, MangledName, FT);
}

heavy::FuncOp OpGen::createSyntaxFunction(SourceLocation Loc,
                                          heavy::Value Proc) {
  llvm_unreachable("not used");
  heavy::FuncOp SyntaxFn = createSyntaxFunction(Loc);
  mlir::Block& Body = *SyntaxFn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard IG(Builder);
  Builder.setInsertionPointToStart(&Body);
  TailPosScope TPS(*this);
  IsTailPos = true;
  LambdaScope LS(*this, SyntaxFn);
  mlir::Value Fn = GetSingleResult(Proc);
  auto Args = Body.getArguments().drop_front();
  create<ApplyOp>(Loc, Fn, Args);
  return SyntaxFn;
}

//  - SyntaxSpec is the full <Keyword> <TransformerSpec> pair
//    that is passed to the syntax function.
//  - OrigCall is likely (define-syntax ...) or (let-syntax ...)
mlir::Value OpGen::createSyntaxSpec(Pair* SyntaxSpec, Value OrigCall) {
  mlir::Value Result;
  Environment* TopLevelEnv = nullptr;
  // Save the insertion point for top level define-syntax.
  mlir::OpBuilder::InsertPoint PrevIp = Builder.saveInsertionPoint();

  Symbol* Keyword = dyn_cast<Symbol>(SyntaxSpec->Car);
  if (!Keyword) return SetError("expecting syntax spec keyword", SyntaxSpec);

  std::string MangledName;
  if (isTopLevel()) {
    TopLevelEnv = cast<Environment>(Context.EnvStack);
    // Create the syntax as a global.
    MangledName = mangleSyntax(Keyword);
    if (MangledName.empty())
      return Error();
    SourceLocation DefineLoc = OrigCall.getSourceLocation();
    // FIXME This code is very similar to stuff in createTopLevelDefine
    auto GlobalOp = createTopLevel<heavy::GlobalOp>(DefineLoc, MangledName);
    setTopLevelOp(GlobalOp.getOperation());
    mlir::Block& Block = *GlobalOp.addEntryBlock();
    // Set insertion point.
    Builder.setInsertionPointToStart(&Block);
  }

  Pair* TransformerSpec = dyn_cast<Pair>(SyntaxSpec->Cdr.car());
  Value SyntaxOperator;

  if (isIdentifier(TransformerSpec->Car))
    SyntaxOperator = Context.GetSyntax(LookupEnv(TransformerSpec->Car)).Value;

  if (!SyntaxOperator)
    return SetError("invalid syntax spec", SyntaxSpec);

  Result = CallSyntax(SyntaxOperator, SyntaxSpec);
  auto SyntaxOp = Result ? Result.getDefiningOp<heavy::SyntaxOp>()
                         : heavy::SyntaxOp();
  if (CheckError())
    return Error();
  if (!SyntaxOp)
    return SetError("expecting syntax transformer", SyntaxSpec);

  if (TopLevelEnv) {
    assert(!MangledName.empty() && "global syntax must have mangled name");
    // Insert a ContOp for the global that was created.
    TopLevelEnv->Insert(Keyword,
        Context.CreateIdTableEntry(MangledName));
    Builder.create<ContOp>(Result.getLoc(), Result);
    Builder.restoreInsertionPoint(PrevIp);
    // Syntax should be available at compile-time.
    Value GlobalOpVal = SyntaxOp->getParentOp();
    Context.RunSync(HEAVY_BASE_VAR(op_eval), GlobalOpVal);
  } else {
    return SetError("TODO support local syntax");
    //Binding* B = Context.CreateBinding(Keyword, Syntax);
    //Context.PushLocalBinding(B);
  }

  return mlir::Value();
}

mlir::Value OpGen::createSyntaxRules(SourceLocation Loc,
                                     Value Keyword,
                                     Value Ellipsis,
                                     heavy::Value LiteralList,
                                     heavy::Value PatternDefs) {
  // Create (anonymous) syntax function.
  heavy::FuncOp FuncOp = createSyntaxFunction(Loc);

  // Expect list of unique literal identifiers.
  llvm::SmallPtrSet<String*, 4> Literals;
  auto insertLiteral = [&](Value V) -> bool {
    // TODO Support bound identifiers for literal list (I think).
    Symbol* S = dyn_cast<Symbol>(V);
    if (!S) {
      SetError("expecting keyword literal", V);
      return true;
    }
    if (equal(V, (Ellipsis))) {
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
  auto SyntaxOp = create<heavy::SyntaxOp>(Loc, FuncOp.getSymName());
  mlir::Block& Body = *FuncOp.addEntryBlock();
  mlir::BlockArgument ExprArg = Body.getArgument(1);
  mlir::BlockArgument EnvArg = Body.getArgument(2);

  mlir::OpBuilder::InsertionGuard IG(Builder);
  Builder.setInsertionPointToStart(&Body);

  // Iterate through each pattern/template pair.
  assert(!CheckError() && "should not have errors here");
  while (Pair* I = dyn_cast<Pair>(PatternDefs)) {
    heavy::Pair* X        = dyn_cast<Pair>(I->Car);
    heavy::Value Pattern  = X->Car;
    heavy::Value Template = X->Cdr.car();
    bool IsProperPT = isa_and_nonnull<Empty>(X->Cdr.cdr());
    if (!IsProperPT) {
      return SetError("expecting pattern template pair", X);
    }

    auto PatternOp = create<heavy::PatternOp>(Pattern.getSourceLocation());
    mlir::Block& B = PatternOp.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&B);

    Value PrevEnvStack = Context.EnvStack;
    PatternTemplate PT(*this, Keyword, Ellipsis, EnvArg, Literals);

    PT.VisitPatternTemplate(Pattern, Template, ExprArg);
    PatternDefs = I->Cdr;
    // Restore the environment.
    Context.EnvStack = PrevEnvStack;
  }

  // Terminate with error call for when no patterns match.
  mlir::Value ErrorMsg = createLiteral(Loc,
      Context.CreateString("no matching pattern for syntax"));
  std::array<mlir::Value, 2> ErrorArgs{ErrorMsg, ExprArg};
  createSyntaxError(Loc, ErrorArgs);

  if (!isa<Empty>(PatternDefs) || Body.empty()) {
    return SetError("expecting list of pattern templates pairs", PatternDefs);
  }
  return SyntaxOp.getResult();
}

// Create a sequence of operations in the current block.
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
        Rest = Current->Car;
        break;
      }
      Visit(Current->Car);
      if (CheckError())
        return Error();
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
bool OpGen::WalkDefineInits(Value Env, IdSet& LocalIds) {
  Pair* P = dyn_cast<Pair>(Env);
  if (!P) return false;
  if (isa<EnvFrame>(P->Car)) return false;
  if (WalkDefineInits(P->Cdr, LocalIds)) return true;

  // Check for duplicate names.
  Binding* B = cast<Binding>(P->Car);
  Value Id = B->getIdentifier();
  bool IsNameInserted;
  std::tie(std::ignore, IsNameInserted) =
    LocalIds.insert(getIdentifierUniqueId(Id));
  if (!IsNameInserted) {
    SetError("local variable has duplicate definitions", Id);
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
  IdSet LocalIds;
  return WalkDefineInits(Context.EnvStack, LocalIds);
}

mlir::Value OpGen::createBody(SourceLocation Loc, Value Body) {
  // Flush any internal definitions before starting a new body.
  if (IsLocalDefineAllowed)
    FinishLocalDefines();
  IsLocalDefineAllowed = true;
  return createSequence(Loc, Body);
}

mlir::Value OpGen::createBinding(Binding *B, mlir::Value Init) {
  SourceLocation SymbolLoc = B->getSourceLocation();
  // Bypass the simpler create<Op> overload to avoid triggering
  // the finalization of the local defines.
  mlir::Value BVal = createHelper<BindingOp>(Builder, SymbolLoc, Init);
  BindingTable.insert(B, BVal);

  return BVal;
}

mlir::Value OpGen::createDefine(Value Id, Value DefineArgs,
                                           Value OrigCall) {
  if (isTopLevel()) return createTopLevelDefine(Id, DefineArgs, OrigCall);
  if (!IsLocalDefineAllowed) return SetError("unexpected define", OrigCall);
  // Create the binding with a lazy init.
  // (Include everything after the define
  //  keyword to visit it later because it could
  //  be a terse lambda syntax.)
  Binding* B = Context.CreateBinding(Id, DefineArgs);
  // Push to the local environment.
  Context.PushLocalBinding(B);
  mlir::Value BVal = createBinding(B, mlir::Value());
  assert(IsLocalDefineAllowed && "define should still be allowed");
  return BVal;
}

mlir::Value OpGen::createTopLevelDefine(Value Id, Value DefineArgs,
                                        Value OrigCall) {
  SourceLocation DefineLoc = OrigCall.getSourceLocation();
  if (LibraryEnvProc) {
    return SetError("unexpected define", OrigCall);
  }

  assert(isTopLevel() && "expecting top level");
  Environment* Env = nullptr;
  if (auto* SC = dyn_cast<SyntaxClosure>(Id))
    Env = cast<Environment>(SC->Env);
  else if (CurSyntaxClosure)
    Env = cast<Environment>(CurSyntaxClosure->Env);
  else
    Env = cast<Environment>(Context.EnvStack);

  // Unwrap SyntaxClosures.
  assert(isIdentifier(Id) && "expecting identifier");
  Symbol* S = cast<Symbol>(Context.RebuildLiteral(Id));

  // If EnvStack isn't an Environment then there is local
  // scope information on top of it

  EnvEntry Entry = Env->Lookup(Context, S);
  if (Entry.Value && Entry.MangledName) {
    if (Mangler::isExternalVariable(getModulePrefix(),
                                    Entry.MangledName->getView()))
      return SetError("define overwrites extern global", S);
  }

  // If the name already exists in the current module
  // then it behaves like `set!`
  //auto* Binding = dyn_cast_or_null<heavy::Binding>(Entry.Value);
  if (Entry.MangledName) {
    llvm::StringRef MangledName = Entry.MangledName->getStringRef();
    TailPosScope TPS(*this);
    IsTailPos = false;
    mlir::Value Init = VisitDefineArgs(DefineArgs);
    mlir::Value LocalV = create<LoadGlobalOp>(DefineLoc, MangledName);
    return create<SetOp>(DefineLoc, LocalV, Init);
  }

  heavy::Mangler Mangler(Context);
  std::string MangledName = Mangler.mangleVariable(getModulePrefix(), S);
  if (MangledName.empty())
    return Error();
  // It is possible someone used a variable name not in scope
  // so we created a global and they are defining it now.
  auto GlobalOp = dyn_cast_or_null<heavy::GlobalOp>(
      LookupSymbol(MangledName));
  if (GlobalOp) {
    // Move the GlobalOp to the current ModuleBuilder insert point
    mlir::Operation* Op = GlobalOp.getOperation();
    Op->moveBefore(&Op->getBlock()->back());
  } else {
    GlobalOp = createTopLevel<heavy::GlobalOp>(DefineLoc, MangledName);
  }
  setTopLevelOp(GlobalOp.getOperation());
  mlir::Block& Block = *GlobalOp.addEntryBlock();

  {
    // set insertion point. add initializer
    mlir::OpBuilder::InsertionGuard IG(Builder);
    Builder.setInsertionPointToStart(&Block);

    // Insert into the module level scope
    Env->Insert(S, Context.CreateIdTableEntry(MangledName));
    if (mlir::Value Init = VisitDefineArgs(DefineArgs)) {
      create<ContOp>(DefineLoc, Init);
    }
  }

  return mlir::Value();
}

mlir::Value OpGen::createIf(SourceLocation Loc, Value Cond, Value Then,
                            Value Else) {
  if (IsLocalDefineAllowed)
    FinishLocalDefines();
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

  auto IfContOp = create<heavy::IfContOp>(Loc, CondResult);
  IfContOp.getThenRegion().push_back(ThenBlock);
  IfContOp.getElseRegion().push_back(ElseBlock);

  {
    // Treat then/else regions as tail position since if there is
    // any ApplyOp contained therein the whole operation becomes
    // an IfContOp which is a terminator with its own PushContOp.
    TailPosScope TPS(*this);
    IsTailPos = true;

    auto handleBranch = [&](heavy::Value Value, mlir::Block* Block) {
      LambdaScope LS(*this, IfContOp);
      mlir::OpBuilder::InsertionGuard IG(Builder);
      Builder.setInsertionPointToStart(Block);
      mlir::Value Result = Visit(Value);
      Block = Builder.getBlock();
      if (Result) {
        assert(Block->empty() ||
          !Block->back().hasTrait<mlir::OpTrait::IsTerminator>());
        create<ContOp>(Loc, Result);
      }
    };

    handleBranch(Then, ThenBlock);
    handleBranch(Else, ElseBlock);
  }

  if (TailPos) return mlir::Value();
  return createContinuation(IfContOp);
}

// LHS can be a symbol or a binding
mlir::Value OpGen::createSet(SourceLocation Loc, Value LHS,
                                                 Value RHS) {
  if (IsLocalDefineAllowed)
    FinishLocalDefines();
  TailPosScope TPS(*this);
  IsTailPos = false;
  assert((isa<Binding, Symbol, ExternName, SyntaxClosure>(LHS)) &&
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
  mlir::Value Result;
  if (Pair* LambdaSpec = dyn_cast<Pair>(P->Car)) {
    // We already checked the name.
    Value Id = LambdaSpec->Car;
    assert(isIdentifier(Id) && "expecting identifier");
    Value Formals = LambdaSpec->Cdr;
    Value Body = P->Cdr;
    llvm::StringRef Name = isTopLevel() ? Id.getStringRef()
                                        : llvm::StringRef();
    Result = createLambda(Formals, Body,
                          Id.getSourceLocation(),
                          Name);
  } else if (isIdentifier(P->Car) && isa<Pair>(P->Cdr)) {
    Result = Visit(cast<Pair>(P->Cdr)->Car);
  } else {
    Result = SetError("invalid define syntax", P);
  }

  if (Result && isa<HeavyValueRefsTy>(Result.getType()))
    Result = create<LoadRefOp>(heavy::SourceLocation(), Result, /*Index*/0);

  return Result;
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

  // Wrap the mlir::Operation and use for the lookup.
  auto CanonicalValue = heavy::Value(G);
  auto GlobalOp = cast<heavy::GlobalOp>(G);

  // Localize globals by the GlobalOp's Operation*
  mlir::Value LocalV;
  // If no TopLevelOp exists then no binding scope exists either.
  if (TopLevelOp)
    LocalV = BindingTable.lookup(CanonicalValue);
  if (!LocalV) {
    LocalV = create<LoadGlobalOp>(Loc, GlobalOp.getName());
    BindingTable.insert(CanonicalValue, LocalV);
    return LocalV;
  }

  return LocalizeValue(LocalV, CanonicalValue);
}

mlir::Value OpGen::VisitExternName(ExternName* EN) {
  return createGlobal(EN->getSourceLocation(), EN->getView());
}

mlir::Value OpGen::VisitSyntaxClosure(SyntaxClosure* SC) {
  SyntaxClosure* PrevSC = CurSyntaxClosure;
  CurSyntaxClosure = SC;
  mlir::Value Result = Visit(SC->Node);
  CurSyntaxClosure = PrevSC;

  return Result;
}

mlir::Value OpGen::VisitSymbol(Symbol* S) {
  EnvEntry Entry = LookupEnv(S);
  SourceLocation Loc = S->getSourceLocation();

  if (!Entry) {
    // Default to the name of a global
    heavy::Mangler Mangler(Context);
    std::string MangledName = Mangler.mangleVariable(getModulePrefix(), S);
    if (MangledName.empty())
      return Error();
    return createGlobal(Loc, MangledName);
  }

  return VisitEnvEntry(Loc, Entry);
}

mlir::Value OpGen::VisitEnvEntry(heavy::SourceLocation Loc, EnvEntry Entry) {
  llvm::StringRef MangledName;
  if (Entry.MangledName)
    MangledName = Entry.MangledName->getStringRef();

  if (!MangledName.empty()) {
    return createGlobal(Loc, MangledName);
  }

  return GetSingleResult(Entry.Value);
}

mlir::Value OpGen::VisitBinding(Binding* B) {
  if (auto* E = dyn_cast<ExternName>(B->getValue()))
    return create<LoadGlobalOp>(Context.getLoc(), E->getView());

  mlir::Value V = BindingTable.lookup(B);
  assert(V && "binding must map to a mlir.value");
  return LocalizeValue(V, B);
}

mlir::Value OpGen::HandleCall(Pair* P, heavy::EnvEntry FnEnvEntry) {
  if (IsLocalDefineAllowed)
    FinishLocalDefines();
  heavy::SourceLocation Loc = P->getSourceLocation();
  mlir::Value Fn;
  llvm::SmallVector<mlir::Value, 16> Args;
  {
    TailPosScope TPS(*this);
    IsTailPos = false;

    if (FnEnvEntry)
      Fn = VisitEnvEntry(Loc, FnEnvEntry);
    else
      Fn = GetSingleResult(P->Car);

    if (CheckError())
      return Error();

    Value V = P->Cdr;
    while (auto* P2 = dyn_cast<Pair>(V)) {
      mlir::Value Arg = GetSingleResult(P2->Car);
      if (CheckError())
        return Error();
      if (mlir::Operation* Op = Arg.getDefiningOp()) {
        Op->setLoc(mlir::OpaqueLoc::get(
              P2->getSourceLocation().getOpaqueEncoding(),
              Builder.getContext()));
      }
      Args.push_back(Arg);
      V = P2->Cdr;
    }
    if (!isa<Empty>(V)) {
      return SetError("improper list as call expression", P);
    }
  }
  return createCall(Loc, Fn , Args);
}

mlir::Value OpGen::createCall(heavy::SourceLocation Loc, mlir::Value Fn,
                              llvm::MutableArrayRef<mlir::Value> Args) {
  // Localize all of the operands
  Fn = LocalizeValue(Fn);
  for (mlir::Value& Arg : Args)
    Arg = LocalizeValue(Arg);

  auto Op = create<ApplyOp>(Loc, Fn, Args);

  if (IsTailPos) return mlir::Value();
  return createContinuation(Op);
}

mlir::Value OpGen::createSyntaxError(heavy::SourceLocation Loc,
                               llvm::MutableArrayRef<mlir::Value> Args) {
  // We defer localizing values until instantiation.
  mlir::Value ErrorFn = create<LoadGlobalOp>(Loc, HEAVY_BASE_VAR_STR(error));
  create<ApplyOp>(Loc, ErrorFn, Args);
  return mlir::Value();
}

mlir::Value OpGen::createOpGen(SourceLocation Loc, mlir::Value Input,
                               mlir::Value Env) {
  // OpGenOp should always be considered tail position.
  create<heavy::OpGenOp>(Loc, Input, Env);
  return mlir::Value();
}

// The CallOp is the terminator op that the PushContOp should precede
// in the block.
mlir::Value OpGen::createContinuation(mlir::Operation* CallOp) {
  mlir::FunctionType FT = createFunctionType(/*Arity=*/1,
                                             RestParamKind::ValueRefs);
  std::string MangledName = mangleFunctionName(llvm::StringRef());
  if (MangledName.empty())
    return Error();

  // No source location
  heavy::SourceLocation Loc;

  // Create the continuation's function
  // subsequent operations will be nested within
  // relying on previous insertion guards to pull us out.
  auto F = createFunction(Loc, MangledName, FT);
  PushContinuationScope(F, CallOp);
  mlir::Block* FuncEntry = F.addEntryBlock();
  Builder.setInsertionPointToStart(FuncEntry);

  // Return ValueRefs object.
  return FuncEntry->getArguments()[1];
}

mlir::Value OpGen::CallSyntax(Value Operator, Pair* P) {
  switch (Operator.getKind()) {
    case ValueKind::BuiltinSyntax: {
      Context.setLoc(P->getSourceLocation());
      BuiltinSyntax* BS = cast<BuiltinSyntax>(Operator);
      return BS->Fn(*this, P);
    }
    case ValueKind::Syntax: {
      heavy::Context& Context = this->Context;
      Context.setLoc(P->getSourceLocation());
      Value Input = P;
      Value Env = Context.EnvStack;
      heavy::OpGen* OpGen = Context.OpGen;
      assert(OpGen == this && "sanity check");
      ++RunSyncDepth;
      std::array<heavy::Value, 2> Args{Input, Env};
      Value Result = Context.RunSync(Operator, Args);
      --RunSyncDepth;
      assert(OpGen == Context.OpGen && "OpGen should not unwind itself");
      if (auto ResultErr = dyn_cast_or_null<heavy::Error>(Result))
        SetError(ResultErr);
      return toValue(Result);
    }
    default: {
      llvm_unreachable("expecting syntax");
      return mlir::Value();
    }
  }
}

mlir::Value OpGen::VisitPair(Pair* P) {
  heavy::EnvEntry FnEnvEntry;
  heavy::Value SyntaxOperator;

  if (isIdentifier(P->Car)) {
    FnEnvEntry = LookupEnv(P->Car);
    SyntaxOperator = Context.GetSyntax(FnEnvEntry).Value;
  }

  if (SyntaxOperator)
    return CallSyntax(SyntaxOperator, P);
  else
    return HandleCall(P, FnEnvEntry);
}

mlir::Value OpGen::VisitVector(Vector* V) {
  llvm::SmallVector<mlir::Value, 8> Vals;
  for (auto X : V->getElements()) {
    mlir::Value Val = GetSingleResult(X);
    if (CheckError())
      return mlir::Value();
    Vals.push_back(Val);
  }

  // Localize the values in the current context.
  for (mlir::Value& Val : Vals)
    Val = LocalizeValue(Val);

  return create<heavy::VectorOp>(Context.getLoc(), Vals);
}

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
    Owner = Op->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
  } else {
    mlir::BlockArgument BlockArg = mlir::cast<mlir::BlockArgument>(V);
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

  if (!LS.Op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
    return ParentLocal;

  // Capture V for use in the current function scope.
  mlir::OpBuilder::InsertionGuard IG(Builder);
  auto FuncOp = cast<heavy::FuncOp>(LS.Op);
  mlir::Block& Block = FuncOp.getBody().front();
  Builder.setInsertionPointToStart(&Block);
  LS.Captures.push_back(ParentLocal);
  uint32_t Index = LS.Captures.size() - 1;
  mlir::Value Closure = Block.getArguments().front();
  mlir::Value NewVal = create<LoadRefOp>(Loc, Closure, Index);

  if (B) {
    BindingTable.insertIntoScope(&LS.BindingScope_, B, NewVal);
  }
  return NewVal;
}

mlir::Operation* OpGen::LookupSymbol(llvm::StringRef MangledName) {
  mlir::ModuleOp M = getModuleOp();
  Operation* G = M.lookupSymbol(MangledName);
  if (G) return G;

  // FIXME This needs to account for symbols in sibling modules
  //       I think we should only see imported names
  M = cast<mlir::ModuleOp>(Context.getModuleOp());
  return M.lookupSymbol(MangledName);
}

void OpGen::Export(Value NameList) {
  // Iterate existing ExportIdOp nodes and load them into NameSet
  llvm::SmallSet<llvm::StringRef, 8> NameSet;
  auto& Ops = getModuleOp().getBody()->getOperations();
  for (auto& Op : Ops) {
    auto ExportIdOp = dyn_cast<heavy::ExportIdOp>(Op);
    if (!ExportIdOp) continue;
    auto Result = NameSet.insert(ExportIdOp.getId());
    if (!Result.second) {
      // This is an error in the IR.
      SetError("export has duplicate name");
      return;
    }
  }

  // Now iterate the names and add ExportIdOps.

  Value V = NameList;
  while (Pair* P = dyn_cast<Pair>(V)) {
    Symbol* Source = dyn_cast<Symbol>(P->Car);
    Symbol* Target = Source;
    if (!Target) {
      if (Pair* P2 = dyn_cast<Pair>(P->Car)) {
        if (isSymbol(P2->Car, "rename") && isa<Pair>(P2->Cdr)) {
          Pair* P3 = cast<Pair>(P2->Cdr);
          Source = dyn_cast<Symbol>(P3->Car);
          if (!Source) {
            SetError("expecting identifier", P3);
            return;
          }
          if (Pair* P4 = dyn_cast<Pair>(P3->Cdr)) {
            Target = dyn_cast<Symbol>(P4->Car);
            if (!Target) {
              SetError("expecting identifier", P4);
              return;
            }
            if (!isa<Empty>(P4->Cdr)) {
              SetError("expecting proper list", P4);
              return;
            }
          }
        }
      }
    }
    if (!Source || !Target) {
      SetError("expecting export spec", P);
      return;
    }

    // Check that Target is not a duplicate.
    auto Result = NameSet.insert(Target->getView());
    if (!Result.second) {
      SetError("export has duplicate name", Target);
      return;
    }

    // If the variable or syntax does not exist yet
    // leave out the symbolName to allow binding later.
    // (via createGlobal)
    EnvEntry Entry = LookupEnv(Source);
    llvm::StringRef MangledName = Entry.MangledName
                                ? Entry.MangledName->getStringRef()
                                : llvm::StringRef();

    createHelper<ExportIdOp>(ModuleBuilder, Target->getSourceLocation(),
                             MangledName, Target->getView());

    // Next
    V = P->Cdr;
  }
  if (!isa<Empty>(V)) {
    SetError("expecting proper list");
    return;
  }
  return Context.Cont();
}

heavy::EnvEntry OpGen::LookupEnv(heavy::Value Id) {
  assert(isIdentifier(Id) && "expecting an identifier");
  // For any given SyntaxClosure, use the object as the lookup
  // in the current environment, and then use the raw Symbol
  // in the closed environment.
  heavy::EnvEntry Result;
  heavy::Value ClosedEnv;
  if (auto* SC = dyn_cast<SyntaxClosure>(Id)) {
    ClosedEnv = SC->Env;
    Id = SC->Node;
  } else if (CurSyntaxClosure) {
    ClosedEnv = CurSyntaxClosure->Env;
  }

  assert(isa<Symbol>(Id) && "syntax closure should be unwrapped");

  if (ClosedEnv) {
    SyntaxClosure StackSC;
    Symbol* S = cast<Symbol>(Id);
    StackSC.Env = ClosedEnv;
    StackSC.Node = S;
    Result = Context.Lookup(&StackSC);
    if (!Result)
      Result = Context.Lookup(S, ClosedEnv);
  } else {
    Symbol* S = cast<Symbol>(Id);
    Result = Context.Lookup(S);
  }

  return Result;
}
