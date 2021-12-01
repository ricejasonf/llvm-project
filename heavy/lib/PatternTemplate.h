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

namespace heavy {

// PatternTemplate
//    - Generate code to match patterns and bind pattern variables
//      See R7RS 4.3.2 Pattern Language.
//    - Generate code for templates by visiting them with OpGen with
//      the pattern variables as SyntacticClosures (in TemplateGen)
class PatternTemplate : ValueVisitor<PatternTemplate, mlir::Value> {
  friend ValueVisitor<PatternTemplate, mlir::Value>;
  heavy::OpGen& OpGen;
  Symbol* Keyword;
  Symbol* Ellipsis;
  NameSet& Literals;
  llvm::SmallPtrSet<String*, 4> PatternVars;

  // P - the pattern node
  // E - the input to match

  SourceLocation getLoc() {
    return OpGen.getContext().getLoc();
  }

public:
  PatternTemplate(heavy::OpGen& O,
                  heavy::Symbol* Keyword,
                  heavy::Symbol* Ellipsis,
                  NameSet& Literals)
    : OpGen(O),
      Keyword(Keyword),
      Ellipsis(Ellipsis),
      Literals(Literals),
      PatternVars()
  { }

  // VisitPatternTemplate should be called with OpGen's insertion point in
  // the body of PatternOp
  mlir::Value VisitPatternTemplate(heavy::Value Pattern,
                                   heavy::Value Template, 
                                   mlir::Value E) {
    heavy::SourceLocation Loc = Pattern.getSourceLocation();
    heavy::Context& C = OpGen.getContext();
    if (isa_and_nonnull<Symbol>(Pattern.car())) {
      // Ignore the initial keyword.
      // FIXME We don't actually check name, but other
      //       implementations simply ignore the first
      //       element altogether.
      Pair* P = cast<Pair>(Pattern);
      auto MatchPairOp = OpGen.create<heavy::MatchPairOp>(Loc, E);
      Visit(P->Cdr, MatchPairOp.cdr());
    } else {
      Visit(Pattern, E);
    }

    if (!C.CheckError()) {
      TemplateGen TG(OpGen, PatternVars, Ellipsis);
      TG.VisitTemplate(Template);
    }

    return mlir::Value();
  }

  // returns true if insertion was successful
  mlir::Value bindPatternVar(Symbol* S, mlir::Value E) {
    bool Inserted;
    std::tie(std::ignore, Inserted) = PatternVars.insert(S->getString());
    if (!Inserted) {
      OpGen.SetError(
        "pattern variable name appears in pattern multiple times", S);
    }
    // Create a local variable with a SyntaxClosure of E as the initializer.
    heavy::SourceLocation Loc = S->getSourceLocation();
    auto SynClo = OpGen.create<SyntaxClosureOp>(Loc, E);
    heavy::Context& C = OpGen.getContext();
    Binding* B = C.CreateBinding(S, SynClo.getOperation());
    C.PushLocalBinding(B);
    return OpGen.createBinding(B, SynClo);
  }

  mlir::Value VisitValue(Value P, mlir::Value E) {
    // Disallow nodes that aren't explicitly allowed
    // in the specification. (r7rs 4.3.2)
    return OpGen.SetError("invalid pattern node", P);
  }

  mlir::Value VisitPair(Pair* P, mlir::Value E) {
    // (<pattern>*)
    // (<pattern>* <pattern> <ellipsis> <pattern>*)
    heavy::SourceLocation Loc = P->getSourceLocation();
    auto MatchPairOp = OpGen.create<heavy::MatchPairOp>(Loc, E);

    heavy::Context& C = OpGen.getContext();

    Visit(P->Car, MatchPairOp.car());
    if (!C.CheckError()) {
      Visit(P->Cdr, MatchPairOp.cdr());
    }

    return mlir::Value();
  }

  mlir::Value VisitSymbol(Symbol* P, mlir::Value E) {
    // <underscore>
    if (P->equals("_")) {
      // Since _ always matches anything, there is
      // nothing to check.
      return mlir::Value(); 
    }

    // <pattern identifier> (literal identifier)
    if (Literals.contains(P->getString())) {
      SourceLocation Loc = P->getSourceLocation();
      EnvEntry Entry = OpGen.getContext().Lookup(P);
      if (!Entry) {
        // If the symbol is unbound just use the symbol.
        OpGen.create<heavy::MatchOp>(Loc, P, E);
        return mlir::Value();
      } else {
        // FIXME This will create captures for local syntax which
        //       never need this. Exported syntax always refer to
        //       globals.
        // Match against the binding or instance of value itself.
        mlir::Value PV = OpGen.VisitEnvEntry(Loc, Entry);
        OpGen.create<heavy::MatchIdOp>(Loc, PV, E);
        return mlir::Value();
      }
    }

    // <ellipsis>
    if (P->equals(Ellipsis)) {
      return OpGen.SetError("<ellipsis> is not a valid pattern", P);
    }

    // everything else is a pattern variable
    return bindPatternVar(P, E);
  }

  // Note: MatchOp is an implicitly chaining operation so
  // it has no result.

  mlir::Value VisitEmpty(Empty P, mlir::Value E) {
    OpGen.create<heavy::MatchOp>(getLoc(), P, E);
    return mlir::Value();
  }

  mlir::Value VisitString(String* P, mlir::Value E) {
    // <pattern datum> -> <string>
    OpGen.create<heavy::MatchOp>(getLoc(), P, E);
    return mlir::Value();
  }

  mlir::Value VisitBool(Bool P, mlir::Value E) {
    // <pattern datum> -> <boolean>
    OpGen.create<heavy::MatchOp>(getLoc(), P, E);
    return mlir::Value();
  }

  mlir::Value VisitInt(Int P, mlir::Value E) {
    // <pattern datum> -> <number>
    OpGen.create<heavy::MatchOp>(getLoc(), P, E);
    return mlir::Value();
  }

  mlir::Value VisitFloat(Float* P, mlir::Value E) {
    // <pattern datum> -> <number>
    OpGen.create<heavy::MatchOp>(getLoc(), P, E);
    return mlir::Value();
  }
};

}
