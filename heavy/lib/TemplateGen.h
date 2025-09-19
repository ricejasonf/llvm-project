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
class TemplateGen : TemplateBase<TemplateGen>,
                    ValueVisitor<TemplateGen, TemplateResult> {
  friend TemplateBase<TemplateGen>;
  friend ValueVisitor<TemplateGen, TemplateResult>;

  Symbol* Ellipsis;
  NameSet& PatternVarNames;
  llvm::SmallVectorImpl<mlir::Value>* CurrentPacks = nullptr;

public:
  using ResultTy = TemplateResult;
  using ErrorTy = TemplateError;

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
    ResultTy Result = Visit(Template);

    mlir::Value TransformedSyntax;
    if (mlir::Value* MVP = std::get_if<mlir::Value>(&Result))
      TransformedSyntax = *MVP;
    else
      TransformedSyntax = createLiteral(std::get<heavy::Value>(Result));

    {
      // Move the RenameOps to the end of the PatternOp.
      mlir::Block* Block = OpGen.Builder.getBlock();
      auto InsertionPoint = OpGen.Builder.getInsertionPoint();
      for (mlir::Value RV : RenameOps)
        RV.getDefiningOp()->moveBefore(Block, InsertionPoint);
    }
      
    // Add the RenameOps to an EnvFrame
    // to serve as the base of the environment.
    mlir::Value TemplateEnv = OpGen.create<EnvFrameOp>(Loc, RenameOps);
    OpGen.createOpGen(Loc, TransformedSyntax, TemplateEnv);
  }

private:
  ResultTy VisitValue(Value P) {
    return P;
  }

  mlir::Value ExpandPack(heavy::SourceLocation Loc,
                         heavy::Value Car, heavy::Value Cdr) {
    ResultTy CdrResult = Visit(Cdr);
    auto Body = std::make_unique<mlir::Region>();
    llvm::SmallVector<mlir::Value, 4> Packs;
    llvm::SmallVectorImpl<mlir::Value>* PrevPacks = CurrentPacks;
    CurrentPacks = &Packs;
    mlir::Block& Block = Body->emplaceBlock();
    {
      mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
      OpGen.Builder.setInsertionPointToStart(&Block);
      ResultTy Last = Visit(Car);
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

  ResultTy VisitPair(Pair* P) {
    heavy::SourceLocation Loc = P->getSourceLocation();
    if (auto* P2 = dyn_cast<Pair>(P->Cdr);
        P2 && isa<Symbol>(P2->Car) &&
        cast<Symbol>(P2->Car)->Equiv(Ellipsis)) {
      return ExpandPack(Loc, P->Car, P2->Cdr);
    }

    ResultTy CarResult = Visit(P->Car);
    ResultTy CdrResult = Visit(P->Cdr);

    // If nothing changed
    auto* HCar = std::get_if<heavy::Value>(&CarResult);
    auto* HCdr = std::get_if<heavy::Value>(&CdrResult);
    if (HCar && HCdr && *HCar == P->Car && *HCdr == P->Cdr)
      return P;

    if (OpGen.CheckError())
      return mlir::Value();

    return createCons(Loc, CarResult, CdrResult);
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

  ResultTy VisitSymbol(Symbol* P) {
    if (PatternVarNames.contains(P->getString()))
      return CurrentPacks ? CaptureExpandArg(P) : GetPatternVar(P);

    if (P->Equiv(Ellipsis))
      return OpGen.SetError("unexpected ellipsis", P);

    return createRename(P);
  }
};

}
