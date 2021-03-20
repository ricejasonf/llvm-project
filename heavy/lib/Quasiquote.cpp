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

// TODO maybe represent heavy::Value like this:
//      https://godbolt.org/z/d6GsaY

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

#if 0 // TODO use the new heavy::Value to allow mixing Operations
         with AST in results
class Quasiquoter : private ValueVisitor<Quasiquoter, mlir::Value> {
  friend class ValueVisitor<Quasiquoter, Value>;
  heavy::OpGen& OpGen;
  // Values captured for hygiene purposes
public:

  Quasiquoter(heavy::OpGen& OG)
    : OpGen(OG)
  { }

  // <quasiquotation>
  mlir::Value Run(Pair* P) {
    bool Rebuilt = false;
    // <quasiquotation 1>
    return HandleQuasiquote(P, Rebuilt, /*Depth=*/1);
  }

private:

  mlir::Value createLiteral(Value V) {
    return OpGen.create<LiteralOp>(V->getSourceLocation(), V);
  }

  mlir::Value createSplice(mlir::Value X, mlir::Value Y) {
    llvm_unreachable("TODO");
  }

  mlir::Value VisitValue(Value V, bool& Rebuilt, int Depth) {
    return OpGen.Visit(V);
  }

  // <qq template D>
  mlir::Value HandleQQTemplate(Value V, bool& Rebuilt, int Depth) {
    assert(Depth >= 0 && "Depth should not be negative");
    if (Depth < 1) {
      // Unquoting requires parents to be rebuilt
      Rebuilt = true;
      return OpGen.Visit(V);
    }
    return Visit(V, Rebuilt, Depth);
  }

  // <quasiquotation D>
  mlir::Value HandleQuasiquote(Pair* P, bool& Rebuilt, int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return OpGen.SetError("invalid quasiquote syntax", P);
    mlir::Value  Result = Visit(Input, Rebuilt, Depth);
    if (!Rebuilt) return createLiteral(Input);
    return Result;
  }

  // <unquotation D>
  mlir::Value HandleUnquote(Pair* P, bool& Rebuilt, int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return OpGen.SetError("invalid unquote syntax", P);

    mlir::Value Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) return OpGen.Visit(P);
    return Result;
  }

  mlir::Value HandleUnquoteSplicing(Pair* P, Value Next, bool& Rebuilt,
                                    int Depth) {
    Value Input = GetSingleSyntaxArg(P);
    if (!Input) return OpGen.SetError("invalid unquote-splicing syntax", P);
    mlir::Value Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) return OpGen.Visit(P);

    return createSplice(Result, OpGen.Visit(Next));
  }

  mlir::Value VisitPair(Pair* P, bool& Rebuilt, int Depth) {
    assert(Depth > 0 && "Depth cannot be zero here.");
    heavy::Context& Context = OpGen.getContext();
    if (Context.CheckError()) return OpGen.createUndefined();
    if (P->Car->isSymbol("quasiquote")) {
      return HandleQuasiquote(P, Rebuilt, Depth + 1);
    } else if (P->Car->isSymbol("unquote")) {
      return HandleUnquote(P, Rebuilt, Depth);
    } else if (isa<Pair>(P->Car) &&
               cast<Pair>(P->Car)->Car->isSymbol("unquote-splicing")) {
      Pair* P2 = cast<Pair>(P->Car);
      return HandleUnquoteSplicing(P2, P->Cdr, Rebuilt, Depth);
    } else {
      // Just a regular old pair
      // <list qq template D>
      bool CarRebuilt = false;
      bool CdrRebuilt = false;
      mlir::Value Car = Visit(P->Car, CarRebuilt, Depth);
      mlir::Value Cdr = Visit(P->Cdr, CdrRebuilt, Depth);
      // Portions that are not rebuilt are always literal
      // '<qq template D>
      //if (!CarRebuilt && CdrRebuilt) Car = createLiteral(Car);
      //if (!CdrRebuilt && CarRebuilt) Cdr = createLiteral(Cdr);
      if (!CarRebuilt && CdrRebuilt) 
        assert(isa<LiteralOp>(Car) && "portions that are not rebuilt are always literal");
      if (!CdrRebuilt && CarRebuilt) 
        assert(isa<LiteralOp>(Cdr) && "portions that are not rebuilt are always literal");
      Rebuilt = CarRebuilt || CdrRebuilt;
      if (!Rebuilt) return OpGen.Visit(P);
      SourceLocation Loc = P->getSourceLocation();
      return OpGen.Visit(Context.CreatePairWithSource(Car, Cdr, Loc);
    }
  }

  // TODO VisitVector
};
#endif

}
namespace heavy { namespace builtin_syntax {

mlir::Value quasiquote(OpGen& OG, Pair* P) {
  return mlir::Value();
  //Quasiquoter QQ(OG);
  //return QQ.Run(P);
}

mlir::Value quote(OpGen& OG, Pair* P) {
  Value Arg = GetSingleSyntaxArg(P);
  if (!Arg) return OG.SetError("invalid quote syntax", P);

  return OG.create<LiteralOp>(P->getSourceLocation(), Arg);
}

}}
