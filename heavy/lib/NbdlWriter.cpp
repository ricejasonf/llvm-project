//===------- NbdlWriter.cpp - Classes Nbdl dialect to Cpp -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines NbdlWriter for transpiling Nbdl dialect operations to
//  C++ which tries to maintain source locations to declarations in DSL code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Casting.h"

namespace {
class NbdlWriter {
  // Map mlir values to names within a function scope.
  using ValueMapTy = llvm::ScopedHashTable<mlir::Value, std::string>;
  using Scope = typename ValueMapTy::ScopeTy;

  ValueMapTy ValueMap;
  heavy::Context& Context;
  heavy::Value Err;
  std::string Buffer;
  llvm::raw_string_ostream OS;

  // Track number of members for CreateStoreOp
  // to generate anonymous identifiers if needed.
  unsigned CurrentMemberCount = 0;

  // Copied from OpEval
  static heavy::SourceLocation getSourceLocation(mlir::Location Loc) {
    if (!mlir::isa<mlir::OpaqueLoc>(Loc)) return {};
    return heavy::SourceLocation(
      mlir::OpaqueLoc::getUnderlyingLocation<heavy::SourceLocationEncoding*>(
        mlir::cast<mlir::OpaqueLoc>(Loc)));
  }

  bool CheckError() {
    //  - Returns true if there is an error.
    return Err != Value();
  }

  void SetError(heavy::SourceLocation Loc, llvm::StringRef Msg) {
    assert(!Err && "no squashing errors");
    Err = Context.CreateError(Loc, Str);
  }

  void SetError(llvm::StringRef Msg, mlir::Operation* Op) {
    assert(!Err && "no squashing errors");
    heavy::SourceLocation Loc = getSourceLocation(Op->getLoc());
    Err = Context.CreateError(Loc, Str, heavy::Value(Op));
  }

  void setLocalVar(mlir::Value V, std::string&& Name) {
    assert(ValueMap.count(Name) == 0 && "no shadowing variable names");
    ValueMap.insert(V, std::move(Name));
  }

  void setLocalVar(mlir::Value) {
    // Create anonymous name.
    std::string AnonName = "anon_TODO";
    setLocalVar(mlir::Value, std::move(AnonName));
  }

  void tetLocalVar(mlir::Value, llvm::StringRef Name) {
    // TODO
  }

  void Visit(mlir::Operation* Op) {
    if (CheckError())
      return;

         if (isa<CreateStoreOp>(Op))  return Visit(cast<CreateStoreOp>(Op));
    else if (isa<StoreOp>(Op))        return Visit(cast<StoreOp>(Op));
    else if (isa<VariantOp>(Op))      return Visit(cast<VariantOp>(Op));
    else if (isa<StoreComposeOp>(Op)) return Visit(cast<StoreComposeOp>(Op));
    else if (isa<ConstexprOp>(Op))    return Visit(cast<ConstexprOp>(Op));
    else if (isa<ConstructOp>(Op))    return Visit(cast<ConstructOp>(Op));
    else if (isa<ContOp>(Op))         return Visit(cast<ContOp>(Op));
    else if (isa<FuncOp>(Op))         return Visit(cast<FuncOp>(Op));
    else
      SetError("unhandled operation", heavy::Value(Op));
  }

  void Visit(CreateStoreOp Op) {
    // Skip externally defined stores.
    if (Op.isExternal())
      return;
    assert(!Op.getBody().empty() && "should not have empty body");
    llvm::SmallVector<llvm::StringRef, 8> ConstructArgs;
    llvm_unreachable("TODO");
#if 0
    auto ConstructOp = cast<nbdl::ConstructOp>(Op.getBody().back());
    mlir::Value Input = ConstructOp.getInput();
    assert(isa<mlir::Operation>(Input) && "expecting store as result");
    Visit(Input.getDefiningOp());
#endif
  }

  void Visit(StoreOp Op) {
    llvm::StringRef Name = Op.getName();
    mlir::OperandRange Args = Op.getArgs();
    if (!Args.empty())
      return SetError("TODO implement construction args for StoreOp",
                      heavy::Value(Op.getOperation()));

    llvm_unreachable("TODO");
  }

  void Visit(VariantOp) {
    llvm_unreachable("TODO");
  }

  void Visit(StoreComposeOp Op) {
    llvm_unreachable("TODO FINISH");
    if (isTopLevel()) {
      // Add the RHS as a member.
      assert(isa<heavy::CreateStoreOp>(Op.getParentOp()) &&
             "should be in context of creating a store");

      // Temporarily store anonymous member name if needed.
      std::string AnonMemberName;
      llvm::StringRef MemberName;

      // If the key is not an identifier, then there should
      // be a GetImplOp elsewhere.
      mlir::Attribute KeyAttr = Op.getKey();
      if (auto SymName = dyn_cast<nbdl::SymbolAttr>(KeyAttr))
        MemberName = SymName.getCppIdentifier();

      if (MemberName.empty()) {
        AnonMemberName = std::string("anon__member_") +
          std::to_string(CurrentMemberCount);
        MemberName = llvm::StringRef(AnonMemberName);
      }
      // Create the member constructed with the $rhs.

      ++CurrentMemberCount;
    } else {
      // Create the store_composite type... oof.
    }
  }

  void Visit(ConstexprOp) {
    llvm_unreachable("TODO");
  }

  void Visit(ContOp) {
    llvm_unreachable("TODO");
  }

  void Visit(ConsumerOp Op) {
    llvm_unreachable("TODO");
  }

  void Visit(FuncOp Op) {
    mlir::FunctionType FT = Op.getFunctionType();
    llvm::StringRef Name = Op.getSymName();

    // Write the return type.
    if (FT.getNumResults() > 1)
      return SetError("Function should have less than 2 results.");
    if (FT.getNumResults() == 0)
      OS << "void";
    else
      VisitType(Ft.getResult(0));

    // Write the function name.
    OS << ' ' << Name << '(';

    // Write the parameter list.
    {
      unsigned I = 0;
      for (mlir::Type ParamType : FT.getInputs()) {
        VisitType(ParamType);
        // TODO handle LocalVarNames and add them to scope.
        OS << "arg_" << I;
        ++I;
      }
    }

    OS << ") { ";

    // Print the body.
    mlir::Region& BodyRegion = Op.getBody();
    if (BodyRegion.empty())
      return SetError("empty function body", Op);
    // Assume there is only one block.
    mlir::Block* Block = BodyRegion.begin();
    for (mlir::Operation* Operation : Block)
      Visit(Operation);

    OS << "}";
  }

  Visit(GetOp Op) {
    llvm::StringRef ResultVarName = setLocalVar(Op.getResult());
    OS << "decltype(auto) " <<
          ResultVarName <<
          "nbdl::get(" <<
          getLocalVar(Op.getState()) <<
          ", " <<
          getLocalVar(Op.getKey()) <<
          ");";
  }

  /************************************
   *********** Type Printing **********
   ************************************/

  void VisitType(mlir::Operation* Op) {
         if (isa<CreateStoreOp>(Op))
           return VisitType(cast<CreateStoreOp>(Op));
    else if (isa<LoadStoreOp>(Op))    return VisitType(cast<LoadStoreOp>(Op));
    else if (isa<VariantOp>(Op))      return VisitType(cast<VariantOp>(Op));
    else if (isa<StoreComposeOp>(Op))
          return VisitType(cast<StoreComposeOp>(Op));
    else if (isa<ConstexprOp>(Op))    return VisitType(cast<ConstexprOp>(Op));
    else if (isa<ConstructOp>(Op))    return VisitType(cast<ConstructOp>(Op));
    else
      SetError("unhandled operation (VisitType)", heavy::Value(Op));
  }

  void VisitType(CreateStoreOp Op) {
    OS << Op.getName();
  }

  void VisitType(LoadStoreOp Op) {
    mlir::Value Result = Op.getResult();
    VisitType(Result);
  }

  void VisitType(VariantOp Op) {
    OS << "nbdl::variant<";
    llvm::interleaveComma(Op.getOperands(), OS,
        [&](mlir::Operand const& Operand) {
          mlir::Value V = Operand.get();   
          VisitType(V);
        }, OS);
    OS << ">";
  }

  void VisitType(StoreComposeOp Op) {
    OS << "nbdl::store_composite<";

    // Key
    auto KeyAttr = cast<mlir::TypeAttr>(Op.getKey());
    mlir::Type KeyType = KeyAttr.getValue();
    VisitType(KeyType);
    OS << ", ";

    // LHS, RHS
    VisitType(Op.getLhs());
    OS << ", ";
    VisitType(Op.getLhs());
    OS << ">";
  }

  void VisitType(ConstexprOp Op) {
    mlir::Value Result = Op.getResult();
    VisitType(Result);
  }

  void VisitType(mlir::Value V) {
    mlir::Type = V.getType();
    VisitType(getSourceLocation(V.getLoc()), V.getType());
  }

  void VisitType(mlir::TypeAttr TA) {
    mlir::Type = A.getValue();
    VisitType(getSourceLocation(TA.getLoc()), V.getType());
  }

  void VisitType(heavy::SourceLocation Loc, mlir::Type Type) {
    if (auto OpaqueType = dyn_cast<nbdl::OpaqueType>(Type))
      OS << Type.getCppTypename();
    else if (auto StoreType = dyn_cast<nbdl::StoreType>(Type))
      OS << Type.getCppTypeName();
    else if (auto StoreType = dyn_cast<nbdl::VariantType>(Type))
      OS << Type.getCppTypeName();
    else if (auto TagType = dyn_cast<nbdl::TagType>(Type))
      OS << Type.getCppTypeName();
    else
      SetError(Loc, "unprintable type");
  }

public:
  NbdlWriter()
    : OS(Buffer)
  { }
};
}
