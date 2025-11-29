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
#include <heavy/Builtins.h>
#include <heavy/Context.h>
#include <heavy/OpGen.h>
#include <heavy/Value.h>
#include <heavy/ValueVisitor.h>
#include "llvm/ADT/ScopedHashTable.h"
#include <llvm/Support/Casting.h>

namespace heavy {

// TemplateGen
//    - Substitute a template using the syntactic environment
//      created by a pattern as part of the syntax-rules syntax.
class TemplateGen : TemplateBase<TemplateGen> {
  friend TemplateBase<TemplateGen>;
  friend ValueVisitor<TemplateGen, TemplateResult>;

  Value Keyword;
  Value Ellipsis;
  NameSet& PatternVarNames;

  // Store the inputs to ExpandPacksOp being constructed.
  using PackCaptureInfoTy = std::pair<mlir::Block*,
                  llvm::SmallVectorImpl<mlir::Value>*>;
  llvm::SmallVector<PackCaptureInfoTy, 4> PackStack;

  bool isEllipsis(heavy::Value Id) {
    return equal(Id, Ellipsis);
  }

public:
  using ErrorTy = TemplateError;

  TemplateGen(heavy::OpGen& O, mlir::Value EnvArg, Value Keyword,
              NameSet& PVNames, Value Ellipsis)
    : TemplateBase(O, EnvArg),
      Keyword(Keyword),
      Ellipsis(Ellipsis),
      PatternVarNames(PVNames)
  { }

  // Create operations to transform syntax and compile the result.
  void BuildTemplate(heavy::Value Template) {
    {
      Symbol* KeywordStr = nullptr;
      // FIXME Unwrapping the SC like this is probably not correct.
      if (auto* SC = dyn_cast<SyntaxClosure>(Keyword))
        KeywordStr = cast<Symbol>(SC->Node);
      else
        KeywordStr = cast<Symbol>(Keyword);
      // Add the Keyword (name of the syntax) as a Rename.
      mlir::Operation* CurrentOp = OpGen.Builder.getBlock()->getParentOp();
      auto SyntaxFuncOp = isa<FuncOp>(CurrentOp)
                                      ? cast<FuncOp>(CurrentOp)
                                      : CurrentOp->getParentOfType<FuncOp>();
      llvm::StringRef SyntaxFuncName = SyntaxFuncOp.getSymName();
      // Insert RenameOps before the RenameEnvOp.
      mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
      OpGen.Builder.setInsertionPoint(RenameEnv);
      auto SyntaxOp = OpGen.create<heavy::SyntaxOp>(
                          Keyword.getSourceLocation(), SyntaxFuncName);
      createRename(KeywordStr, SyntaxOp);
    }

    // Transform the syntax.
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
    mlir::Block& Block = Body->emplaceBlock();
    llvm::SmallVector<mlir::Value, 4> Packs;
    PackStack.push_back({&Block, &Packs});

    {
      mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
      OpGen.Builder.setInsertionPointToStart(&Block);
      TemplateResult Last = Visit(Car);
      if (OpGen.CheckError())
        return OpGen.createUndefined();
      if (std::holds_alternative<heavy::Value>(Last))
        return OpGen.SetError("expansion template should contain pack");
      OpGen.create<ResolveOp>(Loc, createValue(Last));
    }

    // Create the op.
    auto EPO = OpGen.create<ExpandPacksOp>(Loc, createValue(CdrResult),
                                           Packs, std::move(Body));
    assert(!EPO.getPacks().empty() && "should have added some packs");
    PackStack.pop_back();
    return EPO.getResult();
  }

  TemplateResult VisitPair(Pair* P) {
    heavy::SourceLocation Loc = P->getSourceLocation();
    auto* P2 = dyn_cast<Pair>(P->Cdr);
    if (P2 && isEllipsis(P2->Car)) {
      return ExpandPack(Loc, P->Car, P2->Cdr);
    } else if (isSymbol(P->Car, "syntax-source-loc")) {
      // Check for syntax-source-loc auxiliary
      // syntax as an unbound literal.
      heavy::Context& Context = OpGen.getContext();
      auto* S = cast<Symbol>(P->Car);
      if (!Context.Lookup(S)) {
        auto* VarName = dyn_cast_or_null<Symbol>(P2->Car);
        if (VarName && PatternVarNames.contains(VarName->getString())) {
          mlir::Value SC = GetPatternVar(VarName);
          return OpGen.create<SourceLocOp>(Loc, SC);
        } else {
          return OpGen.SetError("expecting pattern variable: {}", VarName);
        }
      }
    }

    return TemplateBase::VisitPair(P);
  }

  mlir::Value GetPatternVar(heavy::Symbol* S) {
    if (OpGen.CheckError())
      return OpGen.createUndefined();
    heavy::Context& Context = OpGen.getContext();
    heavy::Value SCV = Context.Lookup(S).Value;
    if (Binding* B = dyn_cast<Binding>(SCV))
      SCV = B->getValue();
    auto* Op = cast<mlir::Operation>(SCV);
    auto SynClo = cast<SyntaxClosureOp>(Op);
    mlir::Value Result = SynClo.getResult();
    unsigned PackDepth = PackStack.size();
    unsigned CurPackDepth = PackDepth;
    unsigned VarDepth = 0;
    while (auto SP = dyn_cast<SubpatternOp>(
              Result.getDefiningOp()->getParentOp())) {
      ++VarDepth;
      if (CurPackDepth < 1)
        return OpGen.SetError(
            "unexpanded pack must appear in "
            "subtemplate followed by ellipsis: {} "
            "(expansion depth = {}) "
            "(variable depth = {})",
            {S, Int(static_cast<int32_t>(PackDepth)),
                Int(static_cast<int32_t>(VarDepth))});
      assert(Result.hasOneUse() &&
          "pattern variable op must have single use");
      mlir::OpOperand& Operand = *(Result.use_begin());
      Result = SP.getPacks()[Operand.getOperandNumber()];

        // Push the result to the list of pack captures (and add a block argument??).
      --CurPackDepth;
    }

    if (CurPackDepth > 0)
      return OpGen.SetError(
          "pattern variable followed by too many ellipsis", S);

    return MaybeCaptureExpandArg(S->getSourceLocation(), Result);
  }

  // Get or create a capture of an argument to pack expansion.
  mlir::Value MaybeCaptureExpandArg(heavy::SourceLocation Loc,
                                    mlir::Value Result) {
    if (PackStack.empty())
      return Result;

    assert(!PackStack.empty() && "pack expansion op requires pack list");
    for (PackCaptureInfoTy PackCaptureInfo : PackStack) {
      auto [Block, CP] = PackCaptureInfo;
      auto Itr = std::find(CP->begin(), CP->end(), Result);
      if (Itr != CP->end()) {
        Result = Block->getArgument(std::distance(Itr, CP->begin()));
      } else {
        CP->push_back(Result);
        mlir::Type HeavyValueT = OpGen.Builder.getType<HeavyValueTy>();
        mlir::Location MLoc = createLoc(Loc);
        Result = Block->addArgument(HeavyValueT, MLoc);
      }
    }
    return Result;
  }

  TemplateResult VisitSymbol(Symbol* P) {
    if (PatternVarNames.contains(P->getString()))
      return GetPatternVar(P);

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
  Transformer(heavy::OpGen& O, mlir::Value EnvArg, bool IsIr)
    : TemplateBase(O, EnvArg),
      IsImplicitRename(IsIr)
  { }
};

}
