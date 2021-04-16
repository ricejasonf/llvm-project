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
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"
#include <deque>
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
  friend class ValueVisitor<OpGen, mlir::Value>;
  using BindingScopeTable = llvm::ScopedHashTable<
                                            heavy::Binding*,
                                            mlir::Value>;
  using BindingScope = typename BindingScopeTable::ScopeTy;

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

    LambdaScope(OpGen& O, mlir::Operation* Op)
      : O(O)
    {
      O.LambdaScopes.emplace_back(Op, O.BindingTable);
    }

    ~LambdaScope()
    {
      O.LambdaScopes.pop_back();
    }
  };

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
  using LambdaScopeIterator = typename std::deque<LambdaScopeNode>
                                ::reverse_iterator;

  heavy::Context& Context;
  mlir::OpBuilder Builder;
  mlir::OpBuilder LocalInits;
  BindingScopeTable BindingTable;
  std::deque<LambdaScopeNode> LambdaScopes;
  mlir::Operation* TopLevel;
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

public:
  explicit OpGen(heavy::Context& C);

  heavy::Context& getContext() { return Context; }

  mlir::ModuleOp getTopLevel();

  mlir::Value VisitTopLevel(Value V) {
    // there should be an insertion point already setup
    // for the module init function
    IsTopLevel = true;
    return Visit(V);
  }

  bool isTopLevel() { return IsTopLevel; }
  bool isTailPos() { return IsTailPos && !IsTopLevel; }
  bool isLocalDefineAllowed();

  std::string generateLambdaName() {
    return std::string("lambda__") +
           std::to_string(LambdaNameCount++);
  }

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
    return create<Op>(Builder, Loc, std::forward<Args>(args)...);
  }

  mlir::Value createBody(SourceLocation Loc, Value Body);
  mlir::Value createSequence(SourceLocation Loc, Value Body);

  mlir::FunctionType createFunctionType(unsigned Arity,
                                        bool HasRestParam);
  mlir::Value createLambda(Value Formals, Value Body,
                           SourceLocation Loc,
                           llvm::StringRef Name = {});

  mlir::Value createBinding(Binding *B, mlir::Value Init);
  mlir::Value createDefine(Symbol* S, Value Args, Value OrigCall);
  mlir::Value createIf(SourceLocation Loc, Value Cond, Value Then,
                       Value Else);
  mlir::Value createTopLevelDefine(Symbol* S, Value Args, Value OrigCall);
  mlir::Value createTopLevelDefine(Symbol* S, mlir::Value Init, Module* M);
  mlir::Value createUndefined();
  mlir::Value createSet(SourceLocation Loc, Value LHS, Value RHS);

  mlir::Value createEqual(Value V1, Value V2); // equal? for pattern matching
  mlir::Value createLiteral(Value V) {
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

  mlir::Value LocalizeValue(heavy::Binding* B, mlir::Value V);
  mlir::Value LocalizeRec(heavy::Binding* B,
                          mlir::Operation* Op,
                          mlir::Operation* Owner,
                          LambdaScopeIterator Itr);
};

}

#endif
