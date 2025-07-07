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

#include <nbdl_gen/Dialect.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h>
#include <memory>
#include <string>
#include <tuple>

namespace heavy {
class SourceLocationEncoding;
}

namespace {
using namespace nbdl_gen;
using llvm::isa;
using llvm::cast;
using llvm::dyn_cast;

class NbdlWriter {
public:
  // Map mlir values to names within a function scope.
  using ValueMapTy = llvm::ScopedHashTable<mlir::Value, llvm::StringRef>;
  using ValueMapScope = typename ValueMapTy::ScopeTy;

  ValueMapTy ValueMap;
  llvm::BumpPtrAllocator StringHeap;
  // Irritant may be nullptr even with error present.
  mlir::Operation* Irritant = nullptr;
  std::string ErrMsg;
  heavy::SourceLocationEncoding* ErrLoc;
  llvm::raw_ostream& OS;

  // Track number of members for CreateStoreOp
  // to generate anonymous identifiers if needed.
  unsigned CurrentMemberCount = 0;
  unsigned CurrentAnonVarCount = 0;

  NbdlWriter(llvm::raw_ostream& OS)
    : OS(OS)
  { }

  bool CheckError() {
    //  - Returns true if there is an error.
    return !ErrMsg.empty();
  }

  void SetError(mlir::Location Loc, llvm::StringRef Msg) {
    assert(ErrMsg.empty() && "no squashing errors");
    ErrMsg = Msg.str();
    ErrLoc = mlir::OpaqueLoc
      ::getUnderlyingLocationOrNull<heavy::SourceLocationEncoding*>(Loc);
  }

  void SetError(llvm::StringRef Msg, mlir::Operation* Op) {
    assert(ErrMsg.empty() && "no squashing errors");

    SetError(Op->getLoc(), Msg);
    Irritant = Op;
  }

  llvm::StringRef GetLocalVar(mlir::Value V) {
    // ScopedHashTable has a weird interface.
    llvm::StringRef Name = ValueMap.lookup(V);
    if (Name.empty()) {
      SetError(V.getLoc(), "Name not in ValueMap");
      return "ERROR";
    }
    return Name;
  }

  llvm::StringRef SetLocalVar(mlir::Value V, llvm::Twine NameTwine) {
    llvm::SmallString<128> NameTemp;
    NameTwine.toVector(NameTemp);
    // Store the string to have a reliable llvm::StringRef.
    llvm::StringRef Name = NameTemp.str().copy(StringHeap);
    ValueMap.insert(V, Name);

    return Name;
  }

  llvm::StringRef SetLocalVar(mlir::Value V, llvm::StringRef Name,
                              unsigned Num) {
    return SetLocalVar(V, llvm::Twine(Name).concat(llvm::Twine(Num)));
  }

  llvm::StringRef SetLocalVar(mlir::Value V) {
    // Create anonymous name for variable.
    return SetLocalVar(V, "anon_", CurrentAnonVarCount++);
  }

  void Visit(mlir::Operation* Op) {
    if (CheckError())
      return;

         if (isa<CreateStoreOp>(Op))  return Visit(cast<CreateStoreOp>(Op));
    else if (isa<StoreOp>(Op))        return Visit(cast<StoreOp>(Op));
    else if (isa<VariantOp>(Op))      return Visit(cast<VariantOp>(Op));
    else if (isa<StoreComposeOp>(Op)) return Visit(cast<StoreComposeOp>(Op));
    else if (isa<ConstexprOp>(Op))    return Visit(cast<ConstexprOp>(Op));
    else if (isa<ContOp>(Op))         return Visit(cast<ContOp>(Op));
    else if (isa<FuncOp>(Op))         return Visit(cast<FuncOp>(Op));
    else
      SetError("unhandled operation", Op);
  }

  void Visit(CreateStoreOp Op) {
    // Skip externally defined stores.
    if (Op.isExternal())
      return;
    assert(!Op.getBody().empty() && "should not have empty body");
    llvm::SmallVector<llvm::StringRef, 8> ConstructArgs;
    llvm_unreachable("TODO");
#if 0
    auto ConstructOp = cast<nbdl_gen::ConstructOp>(Op.getBody().back());
    mlir::Value Input = ConstructOp.getInput();
    assert(isa<mlir::Operation>(Input) && "expecting store as result");
    Visit(Input.getDefiningOp());
#endif
  }

  void Visit(StoreOp Op) {
#if 0
    llvm::StringRef Name = Op.getName();
    mlir::OperandRange Args = Op.getArgs();
    if (!Args.empty())
      return SetError("TODO implement construction args for StoreOp",
                      Op.getOperation());

#endif
    llvm_unreachable("TODO");
  }

  void Visit(VariantOp) {
    llvm_unreachable("TODO");
  }

  void Visit(StoreComposeOp Op) {
    llvm_unreachable("TODO FINISH");
#if 0
    if (isTopLevel()) {
      // Add the RHS as a member.
      assert(isa<nbdl_gen::CreateStoreOp>(Op.getParentOp()) &&
             "should be in context of creating a store");

      // Temporarily store anonymous member name if needed.
      std::string AnonMemberName;
      llvm::StringRef MemberName;

      // If the key is not an identifier, then there should
      // be a GetImplOp elsewhere.
      mlir::Attribute KeyAttr = Op.getKey();
      if (auto SymName = dyn_cast<nbdl_gen::SymbolAttr>(KeyAttr))
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
#endif
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
      return SetError("Function should have less than 2 results.", Op);
    if (FT.getNumResults() == 0)
      OS << "void";
    else
      VisitType(Op.getLoc(), FT.getResult(0));

    // Write the function name.
    OS << ' ' << Name << '(';

    mlir::Region& BodyRegion = Op.getBody();
    if (BodyRegion.empty())
      return SetError("empty function body", Op);

    ValueMapScope Scope(ValueMap);
    mlir::Block& EntryBlock = BodyRegion.front();
    // Write the parameter list.
    {
      unsigned I = 0;
      llvm::interleaveComma(EntryBlock.getArguments(), OS,
          [&](mlir::BlockArgument const& Arg) {
            OS << SetLocalVar(Arg, "arg_", I);
            ++I;
          });
    }

    OS << ") { ";

    // Print the body assuming a single block.
    for (mlir::Operation& Operation : EntryBlock)
      Visit(&Operation);

    OS << "}";
  }

  void Visit(GetOp Op) {
    OS << "decltype(auto) "
       << SetLocalVar(Op.getResult())
       << "nbdl::get("
       << GetLocalVar(Op.getState())
       << ", "
       << GetLocalVar(Op.getKey())
       << ");";
  }

  /************************************
   *********** Type Printing **********
   ************************************/

  void VisitType(mlir::Operation* Op) {
         if (isa<CreateStoreOp>(Op))
           return VisitType(cast<CreateStoreOp>(Op));
    else if (isa<StoreOp>(Op))    return VisitType(cast<StoreOp>(Op));
    else if (isa<VariantOp>(Op))      return VisitType(cast<VariantOp>(Op));
    else if (isa<StoreComposeOp>(Op))
          return VisitType(cast<StoreComposeOp>(Op));
    else if (isa<ConstexprOp>(Op))    return VisitType(cast<ConstexprOp>(Op));
    else
      SetError("unhandled operation (VisitType)", Op);
  }

  void VisitType(CreateStoreOp Op) {
    OS << Op.getName();
  }

  void VisitType(StoreOp Op) {
    OS << Op.getName();
  }

  void VisitType(VariantOp Op) {
    OS << "nbdl::variant<";
    llvm::interleaveComma(Op.getOperands(), OS,
        [&](mlir::Value const& Val) {
          mlir::Value V = Val;
          VisitType(Op.getLoc(), V);
        });
    OS << ">";
  }

  void VisitType(StoreComposeOp Op) {
    OS << "nbdl::store_composite<";

    // Key
    mlir::Type KeyType = Op.getKey();
    VisitType(Op.getLoc(), KeyType);
    OS << ", ";

    // LHS, RHS
    VisitType(Op.getLoc(), Op.getLhs());
    OS << ", ";
    VisitType(Op.getLoc(), Op.getLhs());
    OS << ">";
  }

  void VisitType(ConstexprOp Op) {
    mlir::Value Result = Op.getResult();
    VisitType(Op.getLoc(), Result);
  }

  void VisitType(mlir::Location Loc, mlir::Value V) {
    VisitType(Loc, V.getType());
  }

  void VisitType(mlir::Location Loc, mlir::TypeAttr TA) {
    VisitType(Loc, TA.getValue());
  }

  void VisitType(mlir::Location Loc, mlir::Type Type) {
    SetError(Loc, "unprintable type");
  }
};
}

namespace nbdl_gen {
std::tuple<std::string, heavy::SourceLocationEncoding*, mlir::Operation*>
translate_cpp(llvm::raw_ostream& OS, mlir::Operation* Op) {
  NbdlWriter Writer(OS);
  Writer.Visit(Op);
  return std::make_tuple(std::move(Writer.ErrMsg),
                         Writer.ErrLoc, Writer.Irritant);

}
}
