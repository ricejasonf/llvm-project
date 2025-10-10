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
#include <heavy/Context.h>
#include <heavy/OpGen.h>
#include <heavy/Value.h>
#include <heavy/ValueVisitor.h>
#include <llvm/Support/Casting.h>

namespace heavy {

// TemplateGen
//    - Substitute a template using the syntactic environment
//      created by a pattern as part of the syntax-rules syntax.
class TemplateGen : TemplateBase<TemplateGen> {
  friend TemplateBase<TemplateGen>;
  friend ValueVisitor<TemplateGen, TemplateResult>;

  Value Ellipsis;
  NameSet& PatternVarNames;
  llvm::SmallVectorImpl<mlir::Value>* CurrentPacks = nullptr;

  bool isEllipsis(heavy::Value Id) {
    return equal(Id, Ellipsis);
  }

public:
  using ErrorTy = TemplateError;

  TemplateGen(heavy::OpGen& O, NameSet& PVNames,
              Value Ellipsis)
    : TemplateBase(O),
      Ellipsis(Ellipsis),
      PatternVarNames(PVNames)
  { }

  // Create operations to transform syntax and compile the result.
  void BuildTemplate(heavy::Value Template) {
    heavy::SourceLocation Loc = Template.getSourceLocation();
    mlir::Value TransformedSyntax = transformSyntax(Template);
    finishRenameEnv();
    createOpGen(Loc, TransformedSyntax);
  }

private:
  mlir::Value ExpandPack(heavy::SourceLocation Loc,
                         heavy::Value Car, heavy::Value Cdr) {
    TemplateResult CdrResult = Visit(Cdr);
    auto Body = std::make_unique<mlir::Region>();
    llvm::SmallVector<mlir::Value, 4> Packs;
    llvm::SmallVectorImpl<mlir::Value>* PrevPacks = CurrentPacks;
    CurrentPacks = &Packs;
    mlir::Block& Block = Body->emplaceBlock();
    {
      mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
      OpGen.Builder.setInsertionPointToStart(&Block);
      TemplateResult Last = Visit(Car);
      if (OpGen.CheckError())
        return mlir::Value();
      if (std::holds_alternative<heavy::Value>(Last))
        return OpGen.SetError("expansion template should contain pack");
      OpGen.create<ResolveOp>(Loc, createValue(Last));
    }
    CurrentPacks = PrevPacks;
    // Create the op.
    auto EPO = OpGen.create<ExpandPacksOp>(Loc, createValue(CdrResult),
                                           Packs, std::move(Body));
    return EPO.getResult();
  }

  TemplateResult VisitPair(Pair* P) {
    heavy::SourceLocation Loc = P->getSourceLocation();
    if (auto* P2 = dyn_cast<Pair>(P->Cdr);
        P2 && isEllipsis(P2->Car)) {
      return ExpandPack(Loc, P->Car, P2->Cdr);
    }

    return TemplateBase::VisitPair(P);
  }

  mlir::Value GetPatternVar(heavy::Symbol* S) {
    heavy::Context& Context = OpGen.getContext();
    heavy::Value SCV = Context.Lookup(S).Value;
    if (Binding* B = dyn_cast<Binding>(SCV))
      SCV = B->getValue();
    auto* Op = cast<mlir::Operation>(SCV);
    auto SynClo = cast<SyntaxClosureOp>(Op);
    mlir::Value Result = SynClo.getResult();
    if (auto SP = dyn_cast<SubpatternOp>(SynClo->getParentOp())) {
      assert(Result.hasOneUse() &&
          "pattern variable should be used only once");
      mlir::OpOperand& Operand = *(Result.use_begin());
      return SP.getPacks()[Operand.getOperandNumber()];
    } else {
      return Result;
    }
  }

  // Get or create a capture of an argument to pack expasion.
  mlir::Value CaptureExpandArg(heavy::Symbol* S) {
    assert(CurrentPacks && "pack expansion op requires pack list");
    auto& CP = *CurrentPacks;

    SourceLocation Loc = S->getSourceLocation();
    mlir::Value Var = GetPatternVar(S);
    mlir::Block* Block = OpGen.Builder.getBlock();

    // Check if Var is already captured.
    auto Itr = std::find(CP.begin(), CP.end(), Var);
    if (Itr != CP.end())
      return Block->getArgument(std::distance(Itr, CP.begin()));

    CP.push_back(Var);
    mlir::Type HeavyValueT = OpGen.Builder.getType<HeavyValueTy>();
    mlir::Location MLoc = createLoc(Loc);
    return Block->addArgument(HeavyValueT, MLoc);
  }

  TemplateResult VisitSymbol(Symbol* P) {
    if (PatternVarNames.contains(P->getString()))
      return CurrentPacks ? CaptureExpandArg(P) : GetPatternVar(P);

    if (isEllipsis(P))
      return OpGen.SetError("unexpected ellipsis", P);

    return createRename(P);
  }
};

// Class for more general macro transfomers.
class Transformer : public TemplateBase<Transformer> {
  friend TemplateBase<Transformer>;
  friend ValueVisitor<Transformer, TemplateResult>;

  bool IsImplicitRename = true;

  TemplateResult VisitSymbol(Symbol* P) {
    if (IsImplicitRename)
      return createRename(P);
    else
      return P;
  }

public:
  Transformer(heavy::OpGen& O, bool IsIr)
    : TemplateBase(O),
      IsImplicitRename(IsIr)
  { }
};

}
