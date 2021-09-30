//===-- PatternTemplate.h - Class for Syntax Pattern Matching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Define heavy::PatternTemplate mapping heavy::Value to mlir::Value
//  for user defined syntax transformations via the `syntax-rules` syntax.
//
//===----------------------------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SmallVector.h"

namespace heavy {

// PatternTemplate
//    - Generate code to match pattern and bind pattern variables
//      See R7RS 4.3.2 Pattern Language.
class PatternTemplate : ValueVisitor<PatternTemplate, mlir::Value> {
  heavy::OpGen& OpGen;
  Symbol* Ellipsis;

  //Value Keywords; // literal identifiers (list)
  llvm::SmallPtrSet<String*, 4> Keywords;
  llvm::SmallPtrSet<String*, 4> PatternVars;

  // P - the pattern node
  // E - the input to match

  void insertKeyword(Symbol* S) {
    if (S->equals(Ellipsis)) {
      OG.setError(S->getSourceLocation
    }
    bool Inserted;
    std::tie(std::ignore, Inserted) = Keywords.insert(S->getString());
    if (!Inserted) {
      OG.setError(S->getSourceLocation(),
                  "keyword specified multiple times");
    }
  }

public:
  PatternTemplate(Symbol* Ellipsis,
                  Value Ks)
    : OpGen(OpGen),
      Ellipsis(Ellipsis),
      Keywords(),
      PatternVars()
  {
    for (Symbol* S : Keywords) {
      insertKeyword(S);
    }
  }

  void VisitPatternTemplate(heavy::Pair* Pattern, heavy::Pair* Template) {
    auto PatternOp = OpGen.create<heavy::PatternOp>();
    Block& B = PatternOp.region().emplaceBlock();
    Visit(Pattern);
    // TODO We need another visitor similar to Quasiquoter to create consOps
    //      with the bindings that we create
    // OpGen.Visit(Template);
  }

  // returns true if insertion was successful
  mlir::Value bindPatternVar(Symbol* S, mlir::Value E) {
    bool Inserted;
    std::tie(std::ignore, Inserted) = PatternVars.insert(S->getString())
    if (!Inserted) {
      OG.setError(S->getSourceLocation(),
                  "pattern variable name used multiple times");
    }
    // Create a local variable with a SyntaxClosure of E as the initializer.
    auto SynClo = OpGen->create<SyntaxClosureOp>(E);
    Context.PushLocalBinding(B);
    return createBinding(B, SynClo);
  }

  mlir::Value VisitValue(Value P, mlir::Value E) {
    // Disallow nodes that aren't explicitly allowed
    // in the specification. (r7rs 4.3.2)
    return OG.setError("invalid pattern node");
  }

  mlir::Value VisitPair(Pair* P, mlir::Value E) {
    // (<pattern>*)
    // (<pattern>* <pattern> <ellipsis> <pattern>*)
    heavy::SourceLocation Loc = P->getSourceLocation();
    auto MatchPairOp = OpGen.create<heavy::MatchPairOp>(Loc, E);

    Visit(P->Car, MatchPairOp.car());
    Visit(P->Cdr, MatchPairOp.cdr());

    return mlir::Value();
  }

  mlir::Value VisitSymbol(Symbol* P, mlir::Value E) {
    // <pattern identifier> (literal identifier)
    if (Keywords.contains(P->getString())) {
      // TODO matched keyword so just skip??
      return mlir::Value();
    }

    // <underscore>
    if (P->equals("_")) {
      // TODO I'm pretty sure we can just ignore this
      return mlir::Value(); 
    }

    // <ellipsis>
    if (P->equals(Ellipsis)) {
      return OpGen.setError("<ellipsis> is not a valid pattern");
    }

    // free variables
    EnvEntry Entry = OpGen.Context.Lookup(P);
    if (Entry) {
      return OpGen.VisitEnvEntry(Loc, Entry);
    }

    // everything else is a pattern variable
    return bindPatternVar(P, E);
  }

  // Note: MatchOp is an implicitly chaining operation so
  // it has no result.

  mlir::Value VisitString(String* P, mlir::Value E) {
    // <pattern datum> -> <string>
    OpGen.create<heavy::MatchOp>(P, E);
    return mlir::Value();
  }

  mlir::Value VisitBool(Bool P, mlir::Value E) {
    // <pattern datum> -> <boolean>
    OpGen.create<heavy::MatchOp>(P, E);
    return mlir::Value();
  }

  mlir::Value VisitInt(Int P, mlir::Value E) {
    // <pattern datum> -> <number>
    OpGen.create<heavy::MatchOp>(P, E);
    return mlir::Value();
  }

  mlir::Value VisitFloat(Float* P, mlir::Value E) {
    // <pattern datum> -> <number>
    OpGen.create<heavy::MatchOp>(P, E);
    return mlir::Value();
  }
};

}
