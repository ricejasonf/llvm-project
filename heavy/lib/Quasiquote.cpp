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

#include "heavy/HeavyScheme.h"
#include "heavy/OpGen.h"
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

    if (mlir::Operation* Op = dyn_cast<Operation>(Result)) {
      return Op->getResult(0);
    }
    return createLiteralOp(Result);
  }

private:

  heavy::Value setError(llvm::StringRef S, heavy::Value V) {
    return OpGen.getContext().SetError(S, V);
  }

  heavy::LiteralOp createLiteralOp(Value V) {
    return OpGen.create<LiteralOp>(V.getSourceLocation(), V);
  }

  heavy::Value createLiteral(Value V) {
    // OpGen does not support a LiteralOp of () at the end of
    // a list when it looks at syntax/call expressions. It is
    // extraneous to make an operation wrapper for it anyways.
    if (isa<Empty>(V)) return V;
    return createLiteralOp(V).getOperation();
  }

  mlir::Operation* createList(SourceLocation Loc, mlir::Value X,
                              mlir::Value Y) {
    mlir::Value Empty = createLiteralOp(heavy::Empty{});
    return OpGen.create<ConsOp>(Loc, X,
        OpGen.create<ConsOp>(Loc, Y, Empty));
  }

  mlir::Operation* createSplice(SourceLocation Loc, heavy::Value X,
                                heavy::Value Y) {
    mlir::Operation* OpX = dyn_cast<Operation>(X);
    mlir::Operation* OpY = dyn_cast<Operation>(Y);
    if (!OpX) OpX = createLiteralOp(X);
    if (!OpY) OpY = createLiteralOp(Y);
    return OpGen.create<SpliceOp>(Loc, OpX->getResult(0),
                                       OpY->getResult(0));
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
      // Note that OpGen.Visit is idempotent for operations
      return OpGen.Visit(V).getDefiningOp();
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
    mlir::Operation* Op = dyn_cast<Operation>(Result);
    if (!Op) Op = createLiteralOp(Result);
    auto Loc = P->getSourceLocation();
    return createList(Loc, createLiteralOp(P->Car), Op->getResult(0));
  }

  // <unquotation D>
  heavy::Value HandleUnquote(Pair* P, bool& Rebuilt, int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return setError("invalid unquote syntax", P);

    heavy::Value Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) return P;

    if (Depth == 1) return Result;
    return createList(P->getSourceLocation(),
                      createLiteralOp(P->Car),
                      OpGen.Visit(Result));
  }

  heavy::Value HandleUnquoteSplicing(Pair* P, Value Next, bool& Rebuilt,
                                    int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return setError("invalid unquote-splicing syntax", P);
    heavy::Value Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) return P;
    
    auto Loc = P->getSourceLocation();
    bool NextRebuilt = false;
    heavy::Value NextResult = Visit(Next, NextRebuilt, Depth);
    mlir::Operation *Spliced = createSplice(Loc, Result, NextResult);
    if (Depth == 1) return Spliced;
    return createList(Loc, createLiteralOp(P->Car), Spliced->getResult(0));
  }

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
      return HandleUnquoteSplicing(P2, P->Cdr, Rebuilt, Depth);
    } else {
      // Just a regular old pair
      // <list qq template D>
      bool CarRebuilt = false;
      bool CdrRebuilt = false;
      heavy::Value Car = Visit(P->Car, CarRebuilt, Depth);
      heavy::Value Cdr = Visit(P->Cdr, CdrRebuilt, Depth);

      Rebuilt = CarRebuilt || CdrRebuilt;
      if (!Rebuilt) return P;

      mlir::Value CarVal;
      mlir::Value CdrVal;

      // Portions that are not rebuilt are always literal
      // '<qq template D>
      if (!CarRebuilt && CdrRebuilt) {
        CarVal = createLiteralOp(Car);
        CdrVal = OpGen.Visit(Cdr);
      } else if (!CdrRebuilt && CarRebuilt) {
        CarVal = OpGen.Visit(Car);
        CdrVal = createLiteralOp(Cdr);
      } else {
        CarVal = OpGen.Visit(Car);
        CdrVal = OpGen.Visit(Cdr);
      }

      SourceLocation Loc = P->getSourceLocation();
      return OpGen.create<ConsOp>(Loc, CarVal, CdrVal)
        .getOperation();
    }
  }

  // TODO VisitVector
};

}

namespace heavy { namespace builtin_syntax {

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
