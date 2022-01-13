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
using NameSet = llvm::SmallPtrSetImpl<String*>;

class OpGen : public ValueVisitor<OpGen, mlir::Value> {
  friend ValueVisitor;
  using BindingScopeTable = llvm::ScopedHashTable<
                                            heavy::Value,
                                            mlir::Value>;
  using BindingScope = typename BindingScopeTable::ScopeTy;

  struct LambdaScopeNode {
    mlir::Operation* Op;
    llvm::SmallVector<mlir::Value, 8> Captures;
    BindingScope BindingScope_;

    LambdaScopeNode(mlir::Operation* Op,
          BindingScopeTable& Table)
      : Op(Op),
        Captures(),
        BindingScope_(Table)
    { }
  };

  // LambdaScope - RAII object that pushes an operation that is
  //               FunctionLike to the scope stack along with a
  //               BindingScope where we can insert stack local
  //               values resulting from operations that load from
  //               a global or variable captured by a closure.
  //
  //               Note that BindingScope is still used for
  //               non-isolated scopes (e.g. `let` syntax).
  struct LambdaScope {
    OpGen& O;
    LambdaScopeNode& Node;

    LambdaScope(OpGen& O, mlir::Operation* Op)
      : O(O),
        Node((O.LambdaScopes.emplace_back(Op, O.BindingTable),
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

  // continuation scopes get popped by their containing
  // lambda
  void PushContinuationScope(mlir::Operation* Op) {
    LambdaScopes.emplace_back(Op, BindingTable);
  }

  // pop the scope and build the PushContOp with its captures
  void PopContinuationScope();

  void setTopLevelOp(mlir::Operation* Op) {
    assert(LambdaScopes.size() == 2 &&
        "TopLevelOp should be at module scope with a LambdaScope");
    TopLevelOp = Op;
    LambdaScopes[1].Op = Op;
  }

  using LambdaScopeIterator = typename std::deque<LambdaScopeNode>
                                ::reverse_iterator;

  heavy::Context& Context;

  // ImportsBuilder - Contain insertion point for imported operations
  //                  from Scheme modules.
  mlir::OpBuilder ImportsBuilder;

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

  // IsTopLevelAllowed - Determine if any operations may be inserted
  //                     for use at program level or in a sequence
  //                     within a library definition.
  bool IsTopLevelAllowed = false;
  bool IsLocalDefineAllowed = false;
  std::string ModulePrefix = {};
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
  bool WalkDefineInits(Value Env, NameSet& LocalNames);
  bool FinishLocalDefines();

public:
  explicit OpGen(heavy::Context& C);

  heavy::Context& getContext() { return Context; }

  mlir::ModuleOp getModuleOp();

  // GetSingleResult
  //  - visits a node expecting a single result
  mlir::Value GetSingleResult(heavy::Value V);
  mlir::ValueRange ExpandResults(mlir::Value Result);

  // GetPatternVar - Get a SyntacticClosureOp by its name.
  mlir::Value GetPatternVar(heavy::Symbol* S);

  void setModulePrefix(std::string&& Prefix) {
    ModulePrefix = std::move(Prefix);
  }
  void resetModulePrefix() {
    ModulePrefix.clear();
  }
  llvm::StringRef getModulePrefix() {
    if (ModulePrefix.empty()) {
      return heavy::Mangler::getManglePrefix();
    }
    return ModulePrefix;
  }

  void VisitLibrary(heavy::SourceLocation Loc, std::string&& MangledName,
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

  bool isTopLevel() { return TopLevelOp == nullptr; }
  bool isTailPos() { return IsTailPos; }
  bool isLocalDefineAllowed();

  std::string mangleModule(heavy::Value Name);
  std::string mangleFunctionName(llvm::StringRef Name);

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
    if (Builder.getBlock() == nullptr && IsTopLevelAllowed) {
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
    if (auto OpResult = V.dyn_cast<mlir::OpResult>()) {
      return heavy::Value(OpResult.getOwner());
    }
    if (auto BlockArg = V.dyn_cast<mlir::BlockArgument>()) {
      mlir::Block* B = BlockArg.getOwner();
      return heavy::Value(reinterpret_cast<heavy::ContArg*>(B));
    }
    llvm_unreachable("invalid mlir value kind");
  }

  mlir::Value createOpGen(SourceLocation Loc, mlir::Value Input);
  mlir::Value createBody(SourceLocation Loc, Value Body);
  mlir::Value createSequence(SourceLocation Loc, Value Body);
  mlir::Value createSyntaxSpec(Pair* SyntaxSpec, Value OrigCall);
  mlir::Value createSyntaxRules(SourceLocation Loc, Symbol* Keyword, 
                                Symbol* Ellipsis, Value KeywordList,
                                Value SyntaxDef);
  mlir::Value createIf(SourceLocation Loc, Value Cond, Value Then,
                            Value Else);
  mlir::Value createContinuation(mlir::Region& initCont);

  mlir::FunctionType createFunctionType(unsigned Arity,
                                        bool HasRestParam);
  mlir::Value createLambda(Value Formals, Value Body,
                               SourceLocation Loc,
                               llvm::StringRef Name = {});

  mlir::Value createGlobal(SourceLocation Loc, llvm::StringRef MangledName);
  mlir::Value createBinding(Binding *B, mlir::Value Init);
  mlir::Value createDefine(Symbol* S, Value Args, Value OrigCall);
  mlir::Value createTopLevelDefine(Symbol* S, Value Args, Value OrigCall);
  mlir::Value createUndefined();
  mlir::Value createSet(SourceLocation Loc, Value LHS, Value RHS);

  mlir::Value createEqual(Value V1, Value V2); // equal? for pattern matching
  heavy::LiteralOp createLiteral(Value V) {
    return create<LiteralOp>(V.getSourceLocation(), V);
  }

  template <typename T>
  mlir::Value SetError(T Str, Value V) {
    Context.SetError(Str, V);
    return Error();
  }

  template <typename T>
  mlir::Value SetError(SourceLocation Loc, T Str, Value V) {
    Context.SetError(Loc, Str, V);
    return Error();
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
  // TODO mlir::Value VisitVector(Vector* V);

  mlir::Value Lookup(heavy::Value V) {
    return BindingTable.lookup(V);
  }

  mlir::Operation* LookupSymbol(llvm::StringRef MangledName);
private:
  mlir::Value HandleCall(Pair* P);
};

}

#endif
