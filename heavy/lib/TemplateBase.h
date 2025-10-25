//===-- TemplateBase.h - Class for Syntax Pattern Matching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Define heavy::TemplateBase mapping heavy::Value to mlir::Value
//  for user defined syntax transformations via the `syntax-rules` syntax.
//  This file is provided as header-only to support OpGen.cpp.
//
//===----------------------------------------------------------------------===//

#include <heavy/Context.h>
#include <heavy/OpGen.h>
#include <heavy/Value.h>
#include <heavy/ValueVisitor.h>
#include <mlir/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <variant>

namespace heavy {

// Literals that are not affected by transformations
// can be returned as heavy::Value to prevent having
// to create an operation for contructing every part of
// every literal.
// (ie Keep the LiteralOps as large chunks of scheme AST.)
struct TemplateError { };
using TemplateResult = std::variant<TemplateError, mlir::Value,
                                    heavy::Value>;

// TemplateBase
//    - Provide a base class for quasiquote and syntax-rules
//      templates. This holds the reference to OpGen and provides
//      utility functions for creating operations that build
//      literals interspersed with values.
template <typename Derived>
class TemplateBase : protected ValueVisitor<Derived, TemplateResult> {

public:
  // Template is the input "form" with syntax closures.
  mlir::Value transformSyntax(heavy::Value Template) {
    TemplateResult Result = getDerived().Visit(Template);

    mlir::Value TransformedSyntax;
    if (mlir::Value* MVP = std::get_if<mlir::Value>(&Result))
      TransformedSyntax = *MVP;
    else
      TransformedSyntax = this->createLiteral(std::get<heavy::Value>(Result));
    return TransformedSyntax;
  }

  // Create a new environment with the renamed name/values in it.
  // (They will have their original names in the "renamed environment".)
  mlir::Value finishRenameEnv() {
    heavy::SourceLocation Loc;
    // Create the RenameOps.
    llvm::SmallVector<mlir::Value, 12> RenameOpVector;
    for (auto [Str, Pair] : RenameOps) {
      auto [R, SC] = Pair;
      // Move RenameOps to the beginning of the block.
      R.getDefiningOp()->moveBefore(RenameEnv);
      RenameOpVector.push_back(R);
    }

    // Add the RenameOps to a new RenameEnvOp
    // to serve as the base of the environment.
    mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
    OpGen.Builder.setInsertionPoint(RenameEnv);
    auto NewRenameEnv = OpGen.create<RenameEnvOp>(Loc, RenameEnv.getEnv(),
                                                  RenameOpVector);
    RenameEnv.getResult().replaceAllUsesWith(NewRenameEnv);
    RenameEnv->erase();
    RenameEnv = NewRenameEnv;
    return NewRenameEnv;
  }

  // Compile the result (at the "comile-time"'s "run-time").
  void createOpGen(heavy::SourceLocation Loc, mlir::Value Expr) {
    OpGen.createOpGen(Loc, Expr);
  }


protected:
  friend ValueVisitor<Derived, TemplateResult>;
  using ValueVisitor<Derived, TemplateResult>::getDerived;

  heavy::OpGen& OpGen;
  heavy::RenameEnvOp RenameEnv;
  // Map the results of RenameOps.
  llvm::DenseMap<String*, std::pair<mlir::Value, mlir::Value>> RenameOps;

  TemplateBase(heavy::OpGen& O, mlir::Value EnvArg)
    : OpGen(O),
      RenameEnv(O.create<RenameEnvOp>(heavy::SourceLocation(), EnvArg,
                                      llvm::ArrayRef<mlir::Value>()))
  { }

  mlir::Location createLoc(heavy::SourceLocation Loc) {
    return mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                OpGen.Builder.getContext());
  }

  // createValue - Create a mlir::Value from its stored
  //               representation in heavy::Value or wrap
  //               it in a LiteralOp.
  mlir::Value createValue(TemplateResult Input) {
    if (mlir::Value* MV = std::get_if<mlir::Value>(&Input))
      return *MV;

    if (heavy::Value* V = std::get_if<heavy::Value>(&Input)) {
      mlir::Value MV = OpGen::toValue(*V);
      return MV ? MV : createLiteral(*V);
    }

    // TemplateError
    return OpGen.createUndefined();
  }

  mlir::Value createLiteral(heavy::Value V) {
    return OpGen.create<LiteralOp>(V.getSourceLocation(), V);
  }

  heavy::ConsOp createList(SourceLocation Loc, heavy::Value X,
                              heavy::Value Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    mlir::Value Empty = createLiteral(heavy::Empty{});
    return OpGen.create<ConsOp>(Loc, ValX,
        OpGen.create<ConsOp>(Loc, ValY, Empty));
  }

  mlir::Value createCons(SourceLocation Loc, TemplateResult const& X,
                                             TemplateResult const& Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    return OpGen.create<ConsOp>(Loc, ValX, ValY);
  }

  mlir::Value createSplice(SourceLocation Loc, TemplateResult const& X,
                                               TemplateResult const& Y) {
    mlir::Value ValX = createValue(X);
    mlir::Value ValY = createValue(Y);
    return OpGen.create<SpliceOp>(Loc, ValX, ValY);
  }

  // Specifying ProvidedVal forces the value into the "rename environment".
  mlir::Value createRename(Symbol* P, mlir::Value ProvidedVal = nullptr) {
    // Since RenameOps are memoized, every operation
    // inserted in this function needs to have the FuncOp
    // as its parent operation.
    mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
    String* UniqueStr = P->getString();

    // Check for memoized SC.
    if (!ProvidedVal) {
      auto [R, SC] = RenameOps.lookup(UniqueStr);
      if (SC)
        return SC;
    }

    // The result will be a symbol literal.
    // The built RenameOps are added to the RenameOps list.
    SourceLocation Loc = P->getSourceLocation();
    llvm::StringRef Id = P->getStringRef();

    // Insert Literals and SyntaxClosures after the RenameEnvOp
    OpGen.Builder.setInsertionPointAfter(RenameEnv);

    mlir::Value CaptureVal = ProvidedVal ? ProvidedVal : createLiteral(P);
    mlir::Value SourceVal = CaptureVal;
    mlir::Value SC;
    {
      if (!isa<HeavySyntaxTy>(CaptureVal.getType()))
        SC = OpGen.create<SyntaxClosureOp>(Loc, SourceVal, CaptureVal,
                                           RenameEnv);
    }

    if (ProvidedVal) {
      mlir::Value R = OpGen.create<heavy::RenameOp>(Loc, Id, CaptureVal);
      RenameOps.insert({UniqueStr, {R, SC}});
      return SC;
    }

    // Insert RenameOps before the RenameEnvOp.
    OpGen.Builder.setInsertionPoint(RenameEnv);

    heavy::Context& Context = OpGen.getContext();
    EnvEntry Entry = Context.Lookup(P);

    std::string MangledNameStr;
    llvm::StringRef MangledName;
    if (Entry && !Entry.MangledName) {
      // "Capture" the object and "rename" it
      // We do not localize (aka capture) it until instantiation.
      mlir::Value Capture = OpGen.Lookup(Entry.Value);
      assert(Capture && "expecting value in lookup");
      mlir::Value R = OpGen.create<heavy::RenameOp>(Loc, Id, Capture);
      RenameOps.insert({UniqueStr, {R, SC}});
      return SC;
    } else if (Entry.MangledName) {
      MangledName = Entry.MangledName->getStringRef();
    } else {
      // Default to the name of a global
      heavy::Mangler Mangler(Context);
      MangledNameStr = Mangler.mangleVariable(OpGen.getModulePrefix(), P);
      if (MangledNameStr.empty())
        return OpGen.createUndefined();
      MangledName = llvm::StringRef(MangledNameStr);
    }

    assert(!MangledName.empty() && "should have mangled name");
    mlir::Value R = OpGen.create<RenameGlobalOp>(Loc, Id, MangledName);
    RenameOps.insert({UniqueStr, {R, SC}});
    return SC;
  }

  // ValueVisitor functions

  TemplateResult VisitValue(heavy::Value P) {
    return P;
  }

  TemplateResult VisitSyntaxClosure(SyntaxClosure* SC) {
    mlir::Value EnvVal = any_cast<mlir::Value>(SC->Env);
    if (!EnvVal)
      return OpGen.SetError("expecting env mlir.value");

    if (!isa<Symbol>(SC->Node))
      return OpGen.SetError("expecting a symbol");

    heavy::SourceLocation Loc = SC->Node.getSourceLocation();
    mlir::Value Node = createLiteral(SC->Node);
    mlir::Value SourceVal = Node;
    return OpGen.create<SyntaxClosureOp>(Loc, SourceVal, Node, EnvVal);
  }

  TemplateResult VisitPair(Pair* P) {
    heavy::SourceLocation Loc = P->getSourceLocation();

    TemplateResult CarResult = getDerived().Visit(P->Car);
    TemplateResult CdrResult = getDerived().Visit(P->Cdr);

    // If nothing changed
    auto* HCar = std::get_if<heavy::Value>(&CarResult);
    auto* HCdr = std::get_if<heavy::Value>(&CdrResult);
    if (HCar && HCdr && *HCar == P->Car && *HCdr == P->Cdr)
      return P;

    if (OpGen.CheckError())
      return OpGen.createUndefined();

    return createCons(Loc, CarResult, CdrResult);
  }

  TemplateResult VisitSymbol(Symbol* P) = delete;
};

}
