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
class TemplateBase {
protected:
  heavy::OpGen& OpGen;
  llvm::SmallVector<mlir::Value, 8> RenameOps;

  TemplateBase(heavy::OpGen& O)
    : OpGen(O)
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
    return mlir::Value(); 
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

  mlir::Value createRename(Symbol* P) {
    // The result will be a symbol literal.
    // The built RenameOps are added to the RenameOps list.
    mlir::Value LiteralResult = createLiteral(P);

    heavy::Context& Context = OpGen.getContext();
    EnvEntry Entry = Context.Lookup(P);
    SourceLocation Loc = P->getSourceLocation();
    llvm::StringRef Id = P->getStringRef();

    // Insert RenameOps at the beginning of the pattern block.
    mlir::OpBuilder::InsertionGuard IG(OpGen.Builder);
    OpGen.Builder.setInsertionPointToStart(OpGen.Builder.getBlock());

    std::string MangledNameStr;
    llvm::StringRef MangledName;
    if (Entry && !Entry.MangledName) {
      // "Capture" the object and "rename" it
      // We do not localize (aka capture) it until instantiation.
      mlir::Value Capture = OpGen.Lookup(Entry.Value);
      assert(Capture && "expecting value in lookup");
      mlir::Value R = OpGen.create<heavy::RenameOp>(Loc, Id, Capture);
      RenameOps.push_back(R);
      return LiteralResult;
    } else if (Entry.MangledName) {
      MangledName = Entry.MangledName->getStringRef();
    } else {
      // Default to the name of a global
      heavy::Mangler Mangler(Context);
      MangledNameStr = Mangler.mangleVariable(OpGen.getModulePrefix(), P);
      if (MangledNameStr.empty())
        return mlir::Value();  // Error
      MangledName = llvm::StringRef(MangledNameStr);
    }

    assert(!MangledName.empty() && "should have mangled name");
    mlir::Value R = OpGen.create<RenameGlobalOp>(Loc, Id, MangledName);
    RenameOps.push_back(R);
    return LiteralResult;
  }
};

}
