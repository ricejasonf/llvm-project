//===--Quasiquoter.h - Classes tree evaluation MLIR Operations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::Quasiquoter for the contextual processing of
//  quasiquote syntax
//
//===----------------------------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Casting.h"

namespace {
// GetSingleSyntaxArg
//              - Given a macro expression (keyword datum)
//                return the first argument iff there is only
//                one argument otherwise returns nullptr
//                (same as `cadr`)
heavy::Value GetSingleSyntaxArg(heavy::Pair* P) {
  // P->Car is the syntactic keyword
  heavy::Pair* P2 = llvm::dyn_cast<heavy::Pair>(P->Cdr);
  if (P2 && llvm::isa<heavy::Empty>(P2->Cdr)) {
    return P2->Car;
  }
  return nullptr;
}

}

namespace heavy {

class Quasiquoter : private ValueVisitor<Quasiquoter, heavy::Value> {
  friend class ValueVisitor<Quasiquoter, Value>;
  heavy::OpGen& OpGen;

public:

  Quasiquoter(heavy::OpGen& OG)
    : OpGen(OG)
  { }

  // <quasiquotation>
  mlir::Value Run(heavy::Value Input) {
    bool Rebuilt = false;
    // <quasiquotation 1>
    heavy::Value Result = Visit(Input, Rebuilt, /*Depth=*/1);

    return createValue(Result);
  }

private:
  // createValue - Create a mlir::Value from its stored
  //               representation in heavy::Value or wrap
  //               it in a LiteralOp.
  mlir::Value createValue(heavy::Value V) {
    mlir::Value Val = OpGen::toValue(V);
    if (Val) return OpGen.LocalizeValue(Val);
    return createLiteralOp(V);
  }

  heavy::Value setError(llvm::StringRef S, heavy::Value V) {
    OpGen.getContext().SetError(S, V);
    return Undefined();
  }

  heavy::LiteralOp createLiteralOp(Value V) {
    return OpGen.create<LiteralOp>(V.getSourceLocation(), V);
  }

  mlir::Operation* createList(SourceLocation Loc, heavy::Value X,
                              heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    mlir::Value Empty = createLiteralOp(heavy::Empty{});
    return OpGen.create<ConsOp>(Loc, ValX,
        OpGen.create<ConsOp>(Loc, ValY, Empty));
  }

  mlir::Operation* createCons(SourceLocation Loc, heavy::Value X,
                                                  heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    return OpGen.create<ConsOp>(Loc, ValX, ValY);
  }

  mlir::Operation* createSplice(SourceLocation Loc, heavy::Value X,
                                heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    return OpGen.create<SpliceOp>(Loc, ValX, ValY);
  }

  heavy::Value VisitValue(Value V, bool& Rebuilt, int Depth) {
    return V;
  }

  // <qq template D>
  heavy::Value HandleQQTemplate(Value V, bool& Rebuilt, int Depth) {
    assert(Depth >= 0 && "Depth should not be negative");
    if (Depth < 1) {
      // Unquoting requires parents to be rebuilt
      Rebuilt = true;
      // Note that OpGen.Visit is idempotent for operations and
      // block arguments (ContArg).
      return OpGen::fromValue(OpGen.GetSingleResult(V));
    }
    return Visit(V, Rebuilt, Depth);
  }

  // <quasiquotation D>
  heavy::Value HandleQuasiquote(Pair* P, bool& Rebuilt, int Depth) {
    assert(Depth > 1 && "should be nested depth here");
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return setError("invalid quasiquote syntax", P);
    heavy::Value Result = Visit(Input, Rebuilt, Depth);
    if (!Rebuilt) return P;
    auto Loc = P->getSourceLocation();
    return createList(Loc, P->Car, Result);
  }

  // <unquotation D>
  heavy::Value HandleUnquote(Pair* P, bool& Rebuilt, int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return setError("invalid unquote syntax", P);

    heavy::Value Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) return P;

    if (Depth == 1) return Result;
    return createList(P->getSourceLocation(),
                      P->Car,
                      Result);
  }

  heavy::Value HandleUnquoteSplicing(Pair* Parent, Pair* P, Value Next,
                                     bool& Rebuilt, int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return setError("invalid unquote-splicing syntax", P);
    heavy::Value Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    bool NextRebuilt = false;
    heavy::Value NextResult = Visit(Next, NextRebuilt, Depth);

    // if nothing changed we're done
    if (!Rebuilt && !NextRebuilt) return Parent; 

    // if the unquote-splicing syntax is literal then cons it
    // with the NextResult
    auto Loc = P->getSourceLocation();
    if (Depth > 1) return createCons(Loc,
                                     createList(Loc, P->Car, Result),
                                     NextResult);
    
    mlir::Operation *Spliced = createSplice(Loc, Result, NextResult);
    return Spliced;
  }

  // (<qq template D>*) |
  // (<qq template D>+ . <qq template D>)
  // where D > 0
  heavy::Value HandlePair(Pair* P, bool& Rebuilt, int Depth) {
    assert(Depth > 0 && "Depth cannot be zero here.");

    bool CdrRebuilt = false;
    bool CarRebuilt = false;
    heavy::Value Cdr = Visit(P->Cdr, CdrRebuilt, Depth);
    heavy::Value Car = Visit(P->Car, CarRebuilt, Depth);

    Rebuilt = CarRebuilt || CdrRebuilt;
    if (!Rebuilt) return P;

    SourceLocation Loc = P->getSourceLocation();
    return createCons(Loc, Car, Cdr);
  }

  // <list qq template D> | <unquotation>
  heavy::Value VisitPair(Pair* P, bool& Rebuilt, int Depth) {
    assert(Depth > 0 && "Depth cannot be zero here.");
    heavy::Context& Context = OpGen.getContext();
    if (Context.CheckError()) return Undefined{};
    if (isSymbol(P->Car, "quasiquote")) {
      return HandleQuasiquote(P, Rebuilt, Depth + 1);
    } else if (isSymbol(P->Car, "unquote")) {
      return HandleUnquote(P, Rebuilt, Depth);
    } else if (isa<Pair>(P->Car) &&
               isSymbol(cast<Pair>(P->Car)->Car, "unquote-splicing")) {
      Pair* P2 = cast<Pair>(P->Car);
      return HandleUnquoteSplicing(P, P2, P->Cdr, Rebuilt, Depth);
    } else {
      return HandlePair(P, Rebuilt, Depth);
    }
  }

  // TODO VisitVector
};

}

namespace heavy { namespace base {

mlir::Value quasiquote(OpGen& OG, Pair* P) {
  Value Input = GetSingleSyntaxArg(P);
  if (!Input) OG.SetError("invalid quasiquote syntax", P);
  Quasiquoter QQ(OG);
  // The result is a hybrid of AST/Ops
  return QQ.Run(Input);
}

mlir::Value quote(OpGen& OG, Pair* P) {
  Value Arg = GetSingleSyntaxArg(P);
  if (!Arg) return OG.SetError("invalid quote syntax", P);

  return OG.create<LiteralOp>(P->getSourceLocation(), Arg);
}

}}
