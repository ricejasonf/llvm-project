//===------ OpGen.h - Classes for generating MLIR Operations ----*- C++ -*-===//
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

#ifndef LLVM_HEAVY_OP_GEN_H
#define LLVM_HEAVY_OP_GEN_H

#include "heavy/Context.h"
#include "heavy/Dialect.h"
#include "heavy/Mangle.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h" // function_ref
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include <deque>
#include <string>
#include <utility>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
  class Value;
}

namespace heavy {
using IdSet = std::set<std::pair<uintptr_t, uintptr_t>>;
using NameSet = llvm::SmallPtrSetImpl<String*>;
class PatternTemplate;
class TemplateGen;

template <typename Derived>
class TemplateBase;

class OpGen : public ValueVisitor<OpGen, mlir::Value> {
  friend ValueVisitor;
  friend CopyCollector;
  friend PatternTemplate;
  friend TemplateGen;
  template <typename Derived>
  friend class TemplateBase;
  using BindingScopeTable = llvm::ScopedHashTable<
                                            heavy::Value,
                                            mlir::Value>;
  using BindingScope = typename BindingScopeTable::ScopeTy;

  struct LambdaScope;
  struct LambdaScopeNode {
    mlir::Operation* Op; // The function of the continuation.
    mlir::Operation* CallOp = nullptr;  // The call to insert PushCont before.
    llvm::SmallVector<mlir::Value, 8> Captures;
    BindingScope BindingScope_;

    LambdaScopeNode(mlir::Operation* Op,
                    mlir::Operation* CallOp,
                    BindingScopeTable& Table)
      : Op(Op),
        CallOp(CallOp),
        Captures(),
        BindingScope_(Table)
    { }
  };

  // Continuation scopes get popped by their containing
  // lambda. The PushContOp is inserted right before the
  // specified CallOp.
  void PushContinuationScope(mlir::Operation* Op, mlir::Operation* CallOp) {
    LambdaScopes.emplace_back(Op, CallOp, BindingTable);
  }

  // pop the scope and build the PushContOp with its captures
  void PopContinuationScope();

  void setTopLevelOp(mlir::Operation* Op) {
    assert(LambdaScopes.size() == 1 &&
        "TopLevelOp should be at module scope with a LambdaScope");
    LambdaScopes.emplace_back(Op, nullptr, BindingTable);
    TopLevelOp = Op;
  }

  using LambdaScopeIterator = typename std::deque<LambdaScopeNode>
                                ::reverse_iterator;

  heavy::Context& Context;

  // ModuleBuilder - Contain insertion point for top level operations
  //                 in the output ModuleOp
  mlir::OpBuilder ModuleBuilder;

  // Builder - Contain the current insertion point for non-top-level
  //           operations.
  mlir::OpBuilder Builder;

  BindingScopeTable BindingTable;
  std::deque<LambdaScopeNode> LambdaScopes;

  // ModuleOp is the current output module for a Program/Library
  // It is always nested within the topmost module which also
  // contains top-level operations imported from Scheme modules.
  mlir::Operation* ModuleOp;

  // TopLevelOp The current top level operation being generated.
  //            It may be either CommandOp, GlobalOp, nullptr.
  mlir::Operation* TopLevelOp = nullptr;

  // TopLevelHandler - Use to evaluate potentially multiple top level
  //                   operations. The handler should receive a single
  //                   Value that is an Operation.
  //                   TODO This needs to be visited during GC.
  Value TopLevelHandler = nullptr;

  // LibraryEnvProc - In the context of define-library we allow <library spec>
  //                  to enter the library environment. No operations should
  //                  be inserted if this is set.
  Binding* LibraryEnvProc = nullptr;

  // The current syntax closure will affect how we perform lookup
  // in LookupEnv.
  SyntaxClosure* CurSyntaxClosure = nullptr;

  struct SyntaxClosureScope;

  // Err - The stored error to indicate that the compiler is in an error state.
  //       (Use CheckError())
  Value Err = nullptr;
  // Track depth of calls to RunSync to allow unwinding errors properly.
  size_t RunSyncDepth = 0;

  bool IsLocalDefineAllowed = false;
  heavy::Symbol* ModulePrefix = nullptr;
  unsigned LambdaNameCount = 0;
  bool IsTailPos = true;

  struct TailPosScope {
    bool& State;
    bool PrevState;

    TailPosScope(heavy::OpGen& OpGen)
      : State(OpGen.IsTailPos),
        PrevState(State)
    { }

    TailPosScope(TailPosScope&) = delete;

    ~TailPosScope() {
      State = PrevState;
    }
  };

  void InsertTopLevelCommandOp(SourceLocation Loc);
  bool WalkDefineInits(Value Env, IdSet& LocalNames);
  bool FinishLocalDefines();

public:
  explicit OpGen(heavy::Context& C, heavy::Symbol* ModulePrefix = nullptr);
  ~OpGen();

  heavy::Context& getContext() { return Context; }

  // CheckError
  //  - Returns true if there is an error or exception
  //  - Builtins will have to check this to stop evaluation
  //    when errors occur
  bool CheckError() const {
    return Err ? true : false;
  }

  void ClearError() { Err = nullptr; }


  mlir::ModuleOp getModuleOp();

  // GetSingleResult
  //  - visits a node expecting a single result
  mlir::Value GetSingleResult(heavy::Value V);

  llvm::StringRef getModulePrefix() {
    if (!ModulePrefix || ModulePrefix->getStringRef().empty()) {
      return heavy::Mangler::getManglePrefix();
    }
    return ModulePrefix->getStringRef();
  }

  void Export(Value NameList);
  void VisitLibrary(heavy::SourceLocation Loc, heavy::Symbol* MangledName,
                    heavy::Value LibraryDecls);
  void VisitTopLevel(Value V);
  void FinishTopLevelOp();
  void VisitTopLevelSequence(Value List);
  void SetTopLevelHandler(Value OnTopLevel) {
    if (isa<Undefined>(OnTopLevel)) {
      TopLevelHandler = nullptr;
    } else {
      TopLevelHandler = OnTopLevel;
    }
  }
  void VisitLibrarySpec(Value V);
  void WithLibraryEnv(Value Thunk);
  bool isLibraryContext() {
    return LibraryEnvProc != nullptr;
  }

  bool isTopLevel() { return TopLevelOp == nullptr; }
  bool isTailPos() { return IsTailPos; }
  bool isLocalDefineAllowed();

  std::string mangleModule(heavy::Value Name);
  std::string mangleFunctionName(llvm::StringRef Name);
  std::string mangleVariable(heavy::Value Name);
  std::string mangleSyntax(heavy::Value Name);

  // createHelper - Facilitate creating an operation with proper source
  //                location information.
  //              - Calling this directly bypasses implicit wrapping in
  //                a top level op as well as finalizing local defines.
  template <typename Op, typename ...Args>
  static Op createHelper(mlir::OpBuilder& Builder, heavy::SourceLocation Loc,
                         Args&& ...args) {
    assert(Builder.getInsertionBlock() != nullptr &&
        "Operation must have insertion point");
    mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                               Builder.getContext());
    return Builder.create<Op>(MLoc, std::forward<Args>(args)...);
  }

  template <typename Op, typename ...Args>
  Op create(heavy::SourceLocation Loc, Args&& ...args) {
    if (Builder.getBlock() == nullptr) {
      InsertTopLevelCommandOp(Loc);
    } else if (IsLocalDefineAllowed) {
      // An error may occur here, but we always want
      // this function to return a valid operation.
      // Just continue until the error is realized.
      FinishLocalDefines();
    }
    return createHelper<Op>(Builder, Loc, std::forward<Args>(args)...);
  }

  template <typename Op, typename ...Args>
  Op createTopLevel(heavy::SourceLocation Loc, Args&& ...args) {
    assert(!LibraryEnvProc &&
        "Should not be in a library spec context");
    return createHelper<Op>(ModuleBuilder, Loc, std::forward<Args>(args)...);
  }

  static mlir::Value toValue(heavy::Value V) {
    if (mlir::Operation* Op = V.get<::heavy::ValueSumType::Operation>()) {
      return VisitOperation(Op);
    }
    if (heavy::ContArg* Arg = V.get<::heavy::ValueSumType::ContArg>()) {
      return VisitContArg(Arg);
    }
    return mlir::Value();
  }

  static heavy::Value fromValue(mlir::Value V) {
    if (!V) return heavy::Value();
    if (auto OpResult = mlir::dyn_cast<mlir::OpResult>(V)) {
      return heavy::Value(OpResult.getOwner());
    }
    if (auto BlockArg = mlir::dyn_cast<mlir::BlockArgument>(V)) {
      // FIXME Do we even get here? Is ContArg ever needed?
      mlir::Block* B = BlockArg.getOwner();
      return heavy::Value(reinterpret_cast<heavy::ContArg*>(B));
    }
    llvm_unreachable("invalid mlir value kind");
  }

  mlir::Value createCall(heavy::SourceLocation Loc, mlir::Value Fn,
                         llvm::MutableArrayRef<mlir::Value> Args);
  mlir::Value createError(heavy::SourceLocation Loc,
                          llvm::MutableArrayRef<mlir::Value> Args);
  mlir::Value createOpGen(SourceLocation Loc, mlir::Value Input,
                          mlir::Value Env);
  mlir::Value createBody(SourceLocation Loc, Value Body);
  mlir::Value createSequence(SourceLocation Loc, Value Body);
  mlir::Value createSyntaxSpec(Pair* SyntaxSpec, Value OrigCall);
  mlir::Value createSyntaxRules(SourceLocation Loc, Value Keyword,
                                Value Ellipsis, Value KeywordList,
                                Value SyntaxDef);
  mlir::Value createIf(SourceLocation Loc, Value Cond, Value Then,
                            Value Else);
  mlir::Value createContinuation(mlir::Operation* CallOp);

  enum class RestParamKind {
    None = 0,
    List,
    ValueRefs,
  };

  mlir::FunctionType createFunctionType(unsigned Arity,
                                        RestParamKind RPK);
  heavy::FuncOp createFunction(SourceLocation Loc,
                               mlir::FunctionType FT,
                               llvm::StringRef MangledName = {});
  heavy::FuncOp createSyntaxFunction(SourceLocation Loc);
  heavy::FuncOp createSyntaxFunction(SourceLocation Loc, heavy::Value Proc);
  mlir::Value createLambda(Value Formals, Value Body,
                           SourceLocation Loc,
                           llvm::StringRef Name = {});
  std::pair<heavy::FuncOp, EnvFrame*>
  createLambdaFunction(Value Formals, SourceLocation Loc,
                       llvm::StringRef Name = {});
  mlir::Value createLambdaBody(heavy::SourceLocation Loc,
                               heavy::Value Body,
                               heavy::FuncOp, EnvFrame* EF);
  mlir::Value createCaseLambda(Pair* FullExpr);

  mlir::Value createGlobal(SourceLocation Loc, llvm::StringRef MangledName);
  mlir::Value createBinding(Binding *B, mlir::Value Init);
  mlir::Value createDefine(Value Id, Value Args, Value OrigCall);
  mlir::Value createTopLevelDefine(Value Id, Value Args, Value OrigCall);
  mlir::Value createUndefined();
  mlir::Value createSet(SourceLocation Loc, Value LHS, Value RHS);

  mlir::Value createEqual(Value V1, Value V2); // equal? for pattern matching

  heavy::LiteralOp createLiteral(heavy::SourceLocation Loc, Value V) {
    return create<LiteralOp>(Loc, V);
  }
  heavy::LiteralOp createLiteral(Value V) {
    return createLiteral(V.getSourceLocation(), V);
  }

  void createLoadModule(SourceLocation Loc, Symbol* MangledName);

  mlir::Value SetError(heavy::Error* NewErr) {
    assert((!Err || Value(NewErr) == Err) && "no squashing errors");
    Err = NewErr;
    Context.setLoc(Err.getSourceLocation());
    if (RunSyncDepth == 0)
      Context.Raise(Err);
    else
      Context.Yield(Err);
    return Error();
  }

  template <typename T>
  mlir::Value SetError(SourceLocation Loc, T Str, Value V = Undefined()) {
    heavy::Error* E = Context.CreateError(Loc, Str, Context.CreatePair(V));
    return SetError(E);
  }

  template <typename T>
  mlir::Value SetError(T Str, Value V = Undefined()) {
    return SetError(V.getSourceLocation(), Str, V);
  }

  mlir::Value Error() {
    return createUndefined();
  }

  mlir::Value LocalizeValue(mlir::Value V, heavy::Value B = nullptr);
  mlir::Value VisitEnvEntry(heavy::SourceLocation Loc, EnvEntry Entry);

  mlir::Value LocalizeRec(heavy::Value B,
                          mlir::Value V,
                          mlir::Operation* Owner,
                          LambdaScopeIterator Itr);

  mlir::Value VisitDefineArgs(Value Args);

  mlir::Value VisitValue(Value V) {
    return createLiteral(V);
  }

  mlir::Value VisitUndefined(Undefined U) {
    return createUndefined();
  }

  // VisitOperation and VisitContArg are both idempotent
  // so they are declared static for reuse in the static
  // value conversion functions.
  static mlir::Value VisitOperation(mlir::Operation* Op) {
    assert(Op->getNumResults() == 1 && "expecting a single value");
    return Op->getResult(0);
  }
  static mlir::Value VisitContArg(heavy::ContArg* Arg) {
    mlir::Block* B = reinterpret_cast<mlir::Block*>(Arg);
    // There should be two args including the closure object.
    assert(B->getNumArguments() == 2 && "expecting a single value");
    return B->getArgument(1);
  }

  mlir::Value VisitBuiltin(Builtin* B) {
    return SetError("internal pointer cannot be made external", B);
  }

  mlir::Value VisitExternName(ExternName* EN);
  mlir::Value VisitSyntaxClosure(SyntaxClosure* SC);
  mlir::Value VisitSymbol(Symbol* S);
  mlir::Value VisitBinding(Binding* B);

  mlir::Value VisitPair(Pair* P);
  mlir::Value VisitVector(Vector* V);

  mlir::Value Lookup(heavy::Value V) {
    return BindingTable.lookup(V);
  }

  mlir::Operation* LookupSymbol(llvm::StringRef MangledName);

  heavy::EnvEntry LookupEnv(heavy::Value Id);
private:
  mlir::Value CallSyntax(Value Operator, Pair* P);
  mlir::Value HandleCall(Pair* P, heavy::EnvEntry FnEnvEntry);
};

}

#endif
