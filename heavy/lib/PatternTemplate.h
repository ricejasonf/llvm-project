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

  mlir::Location createLoc(heavy::SourceLocation Loc) {
    return mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                OpGen.Builder.getContext());
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
    if (isa_and_nonnull<Symbol>(Pattern.car())) {
      // Ignore the initial keyword.
      // FIXME We don't actually check name, but other
      //       implementations simply ignore the first
      //       element altogether.
      Pair* P = cast<Pair>(Pattern);
      auto MatchPairOp = OpGen.create<heavy::MatchPairOp>(Loc, E);
      Visit(P->Cdr, MatchPairOp.getCdr());
    } else {
      Visit(Pattern, E);
    }

    if (!OpGen.CheckError()) {
      TemplateGen TG(OpGen, PatternVars, Ellipsis);
      TG.VisitTemplate(Template);
    }

    return mlir::Value();
  }

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
    return mlir::Value();
  }

  mlir::Value VisitValue(Value P, mlir::Value E) {
    // Disallow nodes that aren't explicitly allowed
    // in the specification. (r7rs 4.3.2)
    return OpGen.SetError("invalid pattern node", P);
  }

  mlir::Value VisitTail(heavy::SourceLocation Loc,
                        Value P, mlir::Value E) {
    // Match the list after the `...`.
    uint32_t Length = 1;

    if (Pair* P2 = dyn_cast<Pair>(P))
      Loc = P2->getSourceLocation();

    heavy::Value Cur = P;
    while (Pair* P2 = dyn_cast<Pair>(Cur)) {
      ++Length;
      Cur = P2->Cdr;
    }
    mlir::Value Tail = OpGen.create<MatchTailOp>(Loc, Length, E);

    // Actually match the tail part.
    Visit(P, Tail);

    // The tail is passed to the SubpatternOp.
    return Tail;
  }

  mlir::Value VisitSubpattern(heavy::SourceLocation Loc,
                              Value P, Value Cdr, mlir::Value E) {
    // The Tail will be used as a sentinel value if it is a pair.
    mlir::Value Tail = VisitTail(Loc, Cdr, E);

    // Visit the subpattern.
    auto Body = std::make_unique<mlir::Region>();
    llvm::SmallVector<mlir::Value, 4> Packs;
    mlir::Block& Block = Body->emplaceBlock();
    {
      mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
      OpGen.Builder.setInsertionPointToStart(&Block);
      mlir::Type HeavyValueT = OpGen.Builder.getType<HeavyValueTy>();
      mlir::Location MLoc = createLoc(Loc);
      mlir::Value BodyArg = Block.addArgument(HeavyValueT, MLoc);
      Visit(P, BodyArg);

      // Create range for pack values by finding the
      // SyntaxClosureOps in the Body region.
      for (mlir::Operation& Op : Block) {
        if (auto SC = dyn_cast<SyntaxClosureOp>(&Op))
          Packs.push_back(SC.getResult());
      }

      // Insert the terminator.
      OpGen.create<ResolveOp>(Loc, Packs);
    }

    // Create the SubpatternOp.
    OpGen.create<heavy::SubpatternOp>(
        Loc, E, Tail, std::move(Body), Packs.size());

    // In the template, the nested syntax closures
    // will be looked up and check if its parent is
    // a subpattern op finding its result.
    return mlir::Value();
  }

  mlir::Value VisitPair(Pair* P, mlir::Value E) {
    // (<pattern>*)
    // (<pattern>* <pattern> <ellipsis> <pattern>*)
    heavy::SourceLocation Loc = P->getSourceLocation();
    if (auto* P2 = dyn_cast<Pair>(P->Cdr);
        P2 && isa<Symbol>(P2->Car) &&
        cast<Symbol>(P2->Car)->equals(Ellipsis)) {
      // P->Car is the subpattern.
      // P2->Cdr is the rest of the pattern after the ...
      VisitSubpattern(Loc, P->Car, P2->Cdr, E);
    } else {
      auto M1 = OpGen.create<MatchPairOp>(Loc, E);
      Visit(P->Car, M1.getCar());
      if (!OpGen.CheckError())
        Visit(P->Cdr, M1.getCdr());
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
    if (P->equals(Ellipsis))
      return OpGen.SetError("<ellipsis> is not a valid pattern", P);

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
