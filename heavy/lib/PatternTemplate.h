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
//  This file is provided as header-only to support OpGen.cpp.
//
//===----------------------------------------------------------------------===//

#include "TemplateGen.h"
#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SmallVector.h"

namespace heavy {

// PatternTemplate
//    - Generate code to match patterns and bind pattern variables
//      See R7RS 4.3.2 Pattern Language.
//    - Generate code for templates by visiting them with OpGen with
//      the pattern variables as SyntacticClosures
class PatternTemplate : ValueVisitor<PatternTemplate, mlir::Value> {
  heavy::OpGen& OpGen;
  Symbol* Ellipsis;
  NameSet& Keywords,
  llvm::SmallPtrSet<String*, 4> PatternVars;

  // P - the pattern node
  // E - the input to match

public:
  PatternTemplate(heavy::OpGen& O,
                  heavy::Symbol* Ellipsis,
                  NameSet& Keywords)
    : OpGen(O),
      Ellipsis(Ellipsis),
      Keywords(Keywords),
      PatternVars()
  { }

  // VisitPatternTemplate should be called with OpGen's insertion point in
  // the body of the function.
  mlir::Value VisitPatternTemplate(heavy::Pair* Pattern, heavy::Pair* Template) {
    mlir::OpBuilder::InsertionGuard IG(Builder);
    auto PatternOp = OpGen.create<heavy::PatternOp>();
    Block& B = PatternOp.region().emplaceBlock();
    Builder.setInsertionPointToStart(B);
    Visit(Pattern);

    TemplateGen TG(OpGen, PatternVars);
    TG.VisitTemplate(Template);

    return mlir::Value();
  }

  // returns true if insertion was successful
  mlir::Value bindPatternVar(Symbol* S, mlir::Value E) {
    bool Inserted;
    std::tie(std::ignore, Inserted) = PatternVars.insert(S->getString())
    if (!Inserted) {
      OG.setError(S->getSourceLocation(),
                  "pattern variable name appears in pattern multiple times");
    }
    // Create a local variable with a SyntaxClosure of E as the initializer.
    auto SynClo = OpGen->create<SyntaxClosureOp>(E);
    Context.PushLocalBinding(B);
    return OpGen.createBinding(B, SynClo);
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
      return mlir::Value();
    }

    // <underscore>
    if (P->equals("_")) {
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
