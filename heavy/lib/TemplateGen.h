//===-- TemplateGen.h - Class for Syntax Pattern Matching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Define heavy::TemplateGen mapping heavy::Value to mlir::Value
//  for user defined syntax transformations via the `syntax-rules` syntax.
//  This file is provided as header-only to support OpGen.cpp.
//
//===----------------------------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SmallVector.h"

namespace heavy {

// TemplateGen
//    - Substitute a template using the syntactic environment
//      created by a pattern as part of the syntax-rules syntax.
class TemplateGen : TemplateBase, ValueVisitor<TemplateGen, heavy::Value> {
  friend ValueVisitor<TemplateGen, heavy::Value>;

  Symbol* Ellipsis;
  NameSet& PatternVarNames;

public:
  TemplateGen(heavy::OpGen& O, NameSet& PVNames,
              Symbol* Ellipsis)
    : Templatebase(O),
      Ellipsis(Ellipsis),
      PatternVarNames(PVNames)
  { }

  // VisitTemplate - Create operations to transform syntax and
  //                 evaluate it.
  mlir::Value VisitTemplate(heavy::Value Template) {
    heavy::SourceLocation Loc = Template.getSourceLocation();
    heavy::Value V = Visit(Template);
    mlir::Value TransformedSyntax = isRebuilt(V) ? toValue(V) :
                                                   createLiteral(Loc, V);
    return OpGen.create<EvalOp>(TransformedSyntax);
  }

private:
  heavy::Value VisitValue(Value P, mlir::Value E) {
    return P;
  }

  heavy::Value VisitPair(Pair* P, mlir::Value E) {
    // Handle syntax as OpGen does.
    heavy::SourceLocation Loc = P->getSourceLocation();
    auto MatchPairOp = OpGen.create<heavy::MatchPairOp>(Loc, E);

    heavy::Value Car = Visit(P->Car, MatchPairOp.car());
    heavy::Value Cdr = Visit(P->Cdr, MatchPairOp.cdr());

    if (!isRebuilt(Car, Cdr)) return P;

    return createCons(Loc, Car, Cdr);
  }

  heavy::Value VisitSymbol(Symbol* S, mlir::Value E) {
    if (!PatternVarNames.contains(S->getString())) {
      return P;
    }
    heavy::Value SC = Context.Lookup(S).Value;
    assert(isa<SyntaxClosure>(SC) && "expecting syntax closure");
    mlir::Value SCV = OpGen.LocalizeValue(BindingTable.lookup(SC), SC);
    assert(SCV && "syntax closure not found in binding table");
    return SCV;
  }
};

}
