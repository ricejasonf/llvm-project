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
#include "llvm/ADT/ScopedHashTable.h"
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

// TODO The visitor functions should return
//      mlir:Operation* instead of mlir::Value
//      since there can be multiple results.
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

  using LambdaScopeIterator = typename std::deque<LambdaScopeNode>
                                ::reverse_iterator;

  heavy::Context& Context;
  mlir::OpBuilder TopLevelBuilder;
  mlir::OpBuilder Builder;
  mlir::OpBuilder LocalInits;
  BindingScopeTable BindingTable;
  std::deque<LambdaScopeNode> LambdaScopes;
  mlir::Operation* TopLevel;
  std::string ModulePrefix = {};
  unsigned LambdaNameCount = 0;
  bool IsTopLevel = false;
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

  void insertTopLevelCommandOp(SourceLocation Loc);
  void walkDefineInits(Value Env);
  heavy::Value transformSyntax(Value V);

public:
  explicit OpGen(heavy::Context& C);

  heavy::Context& getContext() { return Context; }

  mlir::ModuleOp getTopLevel();

  mlir::ValueRange ExpandResults(mlir::Value Result) {
    if (Result.isa<mlir::OpResult>()) {
      return Result.getDefiningOp()->getResults();
    }
    else {
      auto BlockArg = Result.cast<mlir::BlockArgument>();
      return BlockArg.getOwner()->getArguments().drop_front();
    }
  }

  // GetSingleResult
  //  - visits a node expecting a single result
  mlir::Value GetSingleResult(heavy::Value V) {
    mlir::Value Result = Visit(V);
    if (auto BlockArg = Result.dyn_cast<mlir::BlockArgument>()) {
      // the size includes the closure object
      if (BlockArg.getOwner()->getArguments().size() != 2) {
        return SetError("invalid continuation arity", V);
      }
    }
    return Result;
  }

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

  mlir::Operation* VisitTopLevel(Value V);

  bool isTopLevel() { return IsTopLevel; }
  bool isTailPos() { return IsTailPos; }
  bool isLocalDefineAllowed();

  std::string mangleFunctionName(llvm::StringRef Name);

  template <typename Op, typename ...Args>
  static Op create(mlir::OpBuilder& Builder, heavy::SourceLocation Loc,
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
      insertTopLevelCommandOp(Loc);
    }
    return create<Op>(Builder, Loc, std::forward<Args>(args)...);
  }

  mlir::Value createBody(SourceLocation Loc, Value Body);
  mlir::Value createSequence(SourceLocation Loc, Value Body);
  mlir::Value createIf(SourceLocation Loc, Value Cond, Value Then,
                            Value Else);
  mlir::Value createContinuation(mlir::Region& initCont);

  mlir::FunctionType createFunctionType(unsigned Arity,
                                        bool HasRestParam);
  mlir::Value createLambda(Value Formals, Value Body,
                               SourceLocation Loc,
                               llvm::StringRef Name = {});

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

private:
  mlir::Value VisitDefineArgs(Value Args);

  mlir::Value VisitValue(Value V) {
    return createLiteral(V);
  }

  mlir::Value VisitOperation(mlir::Operation* Op) {
    // what if the operation has no results?
    return Op->getResult(0);
  }

  mlir::Value VisitBuiltin(Builtin* B) {
    return create<BuiltinOp>(B->getSourceLocation(), B);
  }

  mlir::Value VisitSymbol(Symbol* S);
  mlir::Value VisitBinding(Binding* B);

  mlir::Value HandleCall(Pair* P);
  void HandleCallArgs(Value V,
                      llvm::SmallVectorImpl<mlir::Value>& Args);

  mlir::Value VisitPair(Pair* P);
  // TODO mlir::Value VisitVector(Vector* V);

  mlir::Value LocalizeValue(heavy::Value B, mlir::Value V);
  mlir::Value LocalizeRec(heavy::Value B,
                          mlir::Operation* Op,
                          mlir::Operation* Owner,
                          LambdaScopeIterator Itr);
};

}

#endif
