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

#include "TemplateBase.h"
#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Casting.h"

namespace heavy {

// TemplateGen
//    - Substitute a template using the syntactic environment
//      created by a pattern as part of the syntax-rules syntax.
class TemplateGen : TemplateBase<TemplateGen>,
                    ValueVisitor<TemplateGen, heavy::Value> {
  friend TemplateBase<TemplateGen>;
  friend ValueVisitor<TemplateGen, heavy::Value>;

  Symbol* Ellipsis;
  NameSet& PatternVarNames;

public:
  TemplateGen(heavy::OpGen& O, NameSet& PVNames,
              Symbol* Ellipsis)
    : TemplateBase(O),
      Ellipsis(Ellipsis),
      PatternVarNames(PVNames)
  { }

  // VisitTemplate - Create operations to transform syntax and
  //                 evaluate it.
  void VisitTemplate(heavy::Value Template) {
    heavy::SourceLocation Loc = Template.getSourceLocation();
    heavy::Value V = Visit(Template);
    mlir::Value TransformedSyntax = isRebuilt(V) ? OpGen::toValue(V) :
                                                   createLiteral(V).result();
    OpGen.createOpGen(Loc, TransformedSyntax);
  }

private:
  heavy::Value VisitValue(Value P) {
    return P;
  }

  heavy::Value VisitPair(Pair* P) {
    heavy::SourceLocation Loc = P->getSourceLocation();
    heavy::Value Car = Visit(P->Car);
    heavy::Value Cdr = Visit(P->Cdr);

    if (!isRebuilt(Car, Cdr)) return P;

    return createCons(Loc, Car, Cdr).getOperation();
  }

  heavy::Value VisitSymbol(Symbol* P) {
    if (PatternVarNames.contains(P->getString())) {
      return OpGen.GetPatternVar(P).getDefiningOp();
    }
    mlir::Value Result = OpGen.VisitSymbol(P);
    assert(Result.isa<mlir::OpResult>() && "expecting operation result");
    return OpGen.Visit(P).getDefiningOp();

    EnvEntry Entry = Context.Lookup(S);
    SourceLocation Loc = S->getSourceLocation();

    if (!Entry) {
      // Insert as literal identifier.
      return P;
    }

    if (Entry.MangledName) {
      // Globals are "effectively renamed" to their
      // linkage names for the sake of template hygiene.
      return Context.CreateExternName(Loc, Entry.MangledName);
    }

    // We can refer to (run-time) locals on the (run-time) stack
    return GetSingleResult(Entry.Value);
  }
};

}
