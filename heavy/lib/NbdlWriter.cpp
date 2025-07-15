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

  llvm::StringRef GetLocalVal(mlir::Value V) {
    // ScopedHashTable has a weird interface.
    llvm::StringRef Name = ValueMap.lookup(V);
    if (Name.empty()) {
      SetError(V.getLoc(), "Name not in ValueMap");
      return "ERROR";
    }
    return Name;
  }

  llvm::StringRef SetLocalVal(mlir::Value V, llvm::Twine Twine) {
    llvm::SmallString<128> NameTemp;
    Twine.toVector(NameTemp);
    // Store the string to have a reliable llvm::StringRef.
    llvm::StringRef Name = NameTemp.str().copy(StringHeap);
    ValueMap.insert(V, Name);

    return Name;
  }

  // Use for variables names to suffix with a unique counter value.
  llvm::StringRef SetLocalVarName(mlir::Value V, llvm::StringRef Name) {
    return SetLocalVal(V,
      llvm::Twine(Name).concat(
          llvm::Twine(CurrentAnonVarCount++)));
  }

  llvm::StringRef SetLocalVarName(mlir::Value V) {
    // Create anonymous name for variable.
    return SetLocalVarName(V, "anon_");
  }

  void WriteForwardedExpr(mlir::Value V) {
    llvm::StringRef Expr = GetLocalVal(V);

    // We do not need to forward literals and junk.
    if (mlir::Operation* Op = V.getDefiningOp()) {
      if (isa<LiteralOp, ConstexprOp>(Op)) {
        OS << Expr;
        return;
      }
    }

    OS << "static_cast<decltype(" << Expr << ")>("
       << Expr
       << ")";
  }

  void WriteParamList(llvm::ArrayRef<mlir::BlockArgument> Args) {
    OS << '(';
    llvm::interleaveComma(Args, OS,
        [&](mlir::BlockArgument const& Arg) {
          OS << "auto&& " << SetLocalVarName(Arg, "arg_");
        });
    OS << ')';
  }

  void Visit(mlir::Operation* Op) {
    if (CheckError())
      return;

         if (isa<CreateStoreOp>(Op))  return Visit(cast<CreateStoreOp>(Op));
    else if (isa<StoreOp>(Op))        return Visit(cast<StoreOp>(Op));
    else if (isa<ApplyOp>(Op))        return Visit(cast<ApplyOp>(Op));
    else if (isa<GetOp>(Op))          return Visit(cast<GetOp>(Op));
    else if (isa<VisitOp>(Op))        return Visit(cast<VisitOp>(Op));
    else if (isa<MatchOp>(Op))        return Visit(cast<MatchOp>(Op));
    else if (isa<OverloadOp>(Op))     return Visit(cast<OverloadOp>(Op));
    else if (isa<MatchIfOp>(Op))      return Visit(cast<MatchIfOp>(Op));
    else if (isa<NoOp>(Op))           return Visit(cast<NoOp>(Op));
    else if (isa<VariantOp>(Op))      return Visit(cast<VariantOp>(Op));
    else if (isa<StoreComposeOp>(Op)) return Visit(cast<StoreComposeOp>(Op));
    else if (isa<ConstexprOp>(Op))    return Visit(cast<ConstexprOp>(Op));
    else if (isa<FuncOp>(Op))         return Visit(cast<FuncOp>(Op));
    else if (isa<LiteralOp>(Op))      return Visit(cast<LiteralOp>(Op));
    else if (isa<MemberNameOp>(Op))   return Visit(cast<MemberNameOp>(Op));
    else
      SetError("unhandled operation", Op);
  }

  void Visit(mlir::Region& R) {
    if (!R.hasOneBlock())
      return SetError("expecting a region with a single block",
                      R.getParentOp());
    for (mlir::Operation& Op : R.front())
      Visit(&Op);
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
    OS << ' ' << Name;

    mlir::Region& Body = Op.getBody();
    if (Body.empty())
      return SetError("empty function body", Op);

    ValueMapScope Scope(ValueMap);
    // Write parameters.
    OS << '(';
    llvm::interleaveComma(Body.getArguments(), OS,
        [&](mlir::BlockArgument const& Arg) {
          OS << "auto&& " << SetLocalVarName(Arg, "arg_");
        });
    OS << ')';

    OS << "{\n";
    Visit(Body);
    OS << '}';
  }

  void Visit(ConstexprOp Op) {
    llvm::StringRef Expr = Op.getExpr();
    if (Expr.empty())
      SetError("expecting expr", Op);
    SetLocalVal(Op.getResult(), llvm::Twine(Expr));
  }

  void Visit(LiteralOp Op) {
    mlir::Attribute Attr = Op.getValue();
    if (auto IA = dyn_cast<mlir::IntegerAttr>(Attr);
        IA &&
        (IA.getType().isIndex() || IA.getType().isSignlessInteger())) {
      SetLocalVal(Op.getResult(), llvm::Twine(IA.getInt()));
    } else if (auto SA = dyn_cast<mlir::StringAttr>(Attr)) {
      SetLocalVal(Op.getResult(), llvm::Twine(llvm::StringRef(SA)));
    } else {
      SetError("unknown literal type", Op);
    }
  }

  void Visit(GetOp Op) {
    auto MemberNameOp = Op.getKey().getDefiningOp<nbdl_gen::MemberNameOp>();
      OS << "decltype(auto) "
         << SetLocalVarName(Op.getResult(), "get_")
         << " = ";
    if (MemberNameOp) {
      WriteForwardedExpr(Op.getState());
      OS << '.' << MemberNameOp.getName()
         << ";\n";
    } else {
      OS << "nbdl::get(";
      WriteForwardedExpr(Op.getState());
      OS << ',';
      WriteForwardedExpr(Op.getKey());
      OS << ");\n";
    }
  }

  void Visit(VisitOp Op) {
    WriteForwardedExpr(Op.getFn());
    OS << '(';
    WriteForwardedExpr(Op.getArg());
    OS << ");\n";
  }

  void Visit(MatchOp Op) {
    OS << "nbdl::match(";
    WriteForwardedExpr(Op.getStore());
    OS << ", ";
    WriteForwardedExpr(Op.getKey());
    OS << ", ";
    OS << "\nboost::hana::overload_linearly(";

    auto& Ops = Op.getOverloads().front().getOperations();
    llvm::interleave(Ops, OS,
        [&](mlir::Operation const& OverloadOp) {
          Visit(const_cast<mlir::Operation*>(&OverloadOp));
        }, ",\n");

    OS << "));\n";
  }

  void Visit(OverloadOp Op) {
    mlir::Region& Body = Op.getBody();
    OS << "[&]";
    // Write parameters.
    OS << '(';
    mlir::BlockArgument& Arg = Body.getArguments().front();
    OS << Op.getType() << ' ' << SetLocalVarName(Arg, "arg_");
    OS << ')';
    OS << "{\n";
    Visit(Body);
    OS << '}';
  }

  void Visit(MatchIfOp Op) {
    mlir::Region& Then = Op.getThenRegion();
    mlir::Region& Else = Op.getElseRegion();
    OS << "if (" << GetLocalVal(Op.getPred()) << '('
       << GetLocalVal(Op.getInput()) << ")) {\n";
    Visit(Then);

    // Check if the the else region is a single MatchIfOp
    // for pretty chaining.
    OS << "} else ";
    if (auto ChainedIfOp = dyn_cast<MatchIfOp>(Else.front().front())) {
      Visit(ChainedIfOp);
    } else {
      OS << "{\n";
      Visit(Op.getElseRegion());
      OS << "}\n";
    }
  }

  void Visit(ApplyOp Op) {
    OS << "decltype(auto) "
       << SetLocalVarName(Op.getResult(), "apply_")
       << " = ";
    // No forwarding stuff here
    OS << GetLocalVal(Op.getFn());
    OS << '(';
    llvm::interleaveComma(Op.getArgs(), OS,
        [&](mlir::Value Val) {
          OS << GetLocalVal(Val);
        });
    OS << ");\n";
  }

  void Visit(MemberNameOp Op) {
    // Member name is meaningless without the parent object
    // so we print it in GetOp.
    // We could implement in MatchOp, but it is a very
    // unlikely use case.
  }

  void Visit(NoOp Op) {
    // Do nothing.
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
