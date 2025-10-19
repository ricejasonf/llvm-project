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
#include <heavy/Source.h>
#include <heavy/Value.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

template <typename Derived>
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
  heavy::SourceLocation CurLoc;

  heavy::LexerWriterFnRef LexerWriter;
  std::string OutputBuffer;
  llvm::raw_string_ostream OS;

  // Track number of members of context
  // to generate anonymous identifiers if needed.
  unsigned CurrentMemberCount = 0;
  unsigned CurrentAnonVarCount = 0;
  NbdlWriter(heavy::LexerWriterFnRef LexerWriter)
    : LexerWriter(LexerWriter),
      OS(OutputBuffer)
  { }

  void SetLoc(mlir::Location MLoc) {
    auto Loc = heavy::SourceLocation(mlir::OpaqueLoc
      ::getUnderlyingLocationOrNull<heavy::SourceLocationEncoding*>(MLoc));
    assert(Loc.isValid() && "expecting valid source location");
    CurLoc = Loc;
  }

  void Flush() {
    if (OutputBuffer.empty())
      return;
    LexerWriter(CurLoc, llvm::StringRef(OutputBuffer));
    OutputBuffer.clear();
  }

  Derived& getDerived() {
    return static_cast<Derived&>(*this);
  }

  bool CheckError() {
    //  - Returns true if there is an error.
    return !ErrMsg.empty();
  }

  void SetError(mlir::Location Loc, llvm::StringRef Msg) {
    if (!ErrMsg.empty())
      return;

    ErrMsg = Msg.str();
    ErrLoc = mlir::OpaqueLoc
      ::getUnderlyingLocationOrNull<heavy::SourceLocationEncoding*>(Loc);
  }

  void SetError(llvm::StringRef Msg, mlir::Operation* Op) {
    if (!ErrMsg.empty())
      return;

    SetError(Op->getLoc(), Msg);
    Irritant = Op;
  }

  void SetErrorV(llvm::StringRef Msg, mlir::Value V) {
    mlir::Operation* Op = V.getDefiningOp();
    if (!Op) {
      mlir::Region* R = V.getParentRegion();
      if (R)
        Op = R->getParentOp();
      assert(Op && "mlir.value has no parent operation");
    }
    SetError(Msg, Op);
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

  /************************************
   *********** Expr Printing **********
   ************************************/

  void WriteExpr(mlir::Value V, bool IsFwd = false) {
    // We do not need to forward literals and junk.
    if (auto Op = V.getDefiningOp<LiteralOp>()) {
      WriteExpr(Op);
    } else if (auto Op = V.getDefiningOp<ConstexprOp>()) {
      WriteExpr(Op);
    } else {
      llvm::StringRef Expr = GetLocalVal(V);
      if (IsFwd) {
        OS << "static_cast<decltype(" << Expr << ")>("
           << Expr
           << ")";
      } else {
        OS << Expr;
      }
    }
  }

  void WriteForwardedExpr(mlir::Value V) {
    WriteExpr(V, /*IsFwd=*/true);
  }

  void WriteExpr(ConstexprOp Op) {
    llvm::StringRef Expr = Op.getExpr();
    if (Expr.empty())
      SetError("expecting expr", Op);
    OS << Expr;
  }

  void WriteExpr(LiteralOp Op) {
    mlir::Attribute Attr = Op.getValue();
    if (auto IA = dyn_cast<mlir::IntegerAttr>(Attr);
        IA &&
        (IA.getType().isIndex() || IA.getType().isSignlessInteger())) {
      OS << IA.getInt();
    } else if (auto SA = dyn_cast<mlir::StringAttr>(Attr)) {
      OS << '"' << llvm::StringRef(SA) << '"';
    } else {
      SetError("unknown literal type", Op);
    }
  }

  /************************************
   *********** Type Printing **********
   ************************************/

  void VisitType(mlir::Value Val) {
    if (mlir::Operation* Op = Val.getDefiningOp()) {
      if (isa<ContextOp>(Op))
        return VisitType(cast<ContextOp>(Op));
      else if (isa<StoreOp>(Op))
        return VisitType(cast<StoreOp>(Op));
      else if (isa<VariantOp>(Op))
        return VisitType(cast<VariantOp>(Op));
      else if (isa<StoreComposeOp>(Op))
        return VisitType(cast<StoreComposeOp>(Op));
      else if (isa<ConstexprOp>(Op))
        return VisitType(cast<ConstexprOp>(Op));
      else
        SetError("unhandled operation (VisitType): {}", Op);
    } else {
      // Handle mlir::BlockArgument.
      // Arguments cannot be `decltype(auto)`
      if (auto OpaqueType = dyn_cast<nbdl_gen::OpaqueType>(Val.getType())) {
        OS << "auto&&";
      } else {
        return SetErrorV("unsupported type for argument", Val);
      }
    }
  }

  void VisitType(ContextOp Op) {
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
          VisitType(V);
        });
    OS << ">";
  }

  void VisitType(StoreComposeOp Op) {
    OS << "nbdl::store_composite<";
    VisitType(Op.getKey());
    OS << ", ";
    VisitType(Op.getLhs());
    OS << ", ";
    VisitType(Op.getRhs());
    OS << ">";
  }

  void VisitType(ConstexprOp Op) {
    VisitType(Op.getResult());
  }

  void VisitType(mlir::Location Loc, mlir::Type Type) {
    if (auto OpaqueType = dyn_cast<nbdl_gen::OpaqueType>(Type))
      OS << "decltype(auto)";
    else
      SetError(Loc, "unsupported type");
  }
};

class FuncWriter : public NbdlWriter<FuncWriter> {
  public:
  using NbdlWriter<FuncWriter>::NbdlWriter;

  void Visit(mlir::Operation* Op) {
    Flush();
    heavy::SourceLocation PrevLoc = CurLoc;
    SetLoc(Op->getLoc());

    if (CheckError()) return;
    auto ScopeExit = llvm::make_scope_exit([this, PrevLoc] {
        Flush();
        CurLoc = PrevLoc;
      });

         if (isa<ApplyOp>(Op))        return Visit(cast<ApplyOp>(Op));
    else if (isa<GetOp>(Op))          return Visit(cast<GetOp>(Op));
    else if (isa<VisitOp>(Op))        return Visit(cast<VisitOp>(Op));
    else if (isa<MatchOp>(Op))        return Visit(cast<MatchOp>(Op));
    else if (isa<OverloadOp>(Op))     return Visit(cast<OverloadOp>(Op));
    else if (isa<MatchIfOp>(Op))      return Visit(cast<MatchIfOp>(Op));
    else if (isa<FuncOp>(Op))         return Visit(cast<FuncOp>(Op));
    else if (isa<MemberNameOp>(Op))   return Visit(cast<MemberNameOp>(Op));
    else if (isa<ConstexprOp, LiteralOp>(Op)) return;
    else if (isa<UnitOp>(Op)) return;
    else if (isa<EmptyOp>(Op))
      return SetError("empty type does not map to c++", Op);
    else
      return SetError("unhandled operation: {}", Op);
  }

  void VisitRegion(mlir::Region& R) {
    if (!R.hasOneBlock())
      return SetError("expecting a region with a single block",
                      R.getParentOp());
    for (mlir::Operation& Op : R.front())
      Visit(&Op);
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
      OS << "decltype(auto)";

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
    VisitRegion(Body);
    OS << '}';
  }

  void Visit(GetOp Op) {
    auto MemberNameOp = Op.getKey().getDefiningOp<nbdl_gen::MemberNameOp>();
      OS << "auto&& "
         << SetLocalVarName(Op.getResult(), "get_")
         << " = ";
    if (MemberNameOp) {
      WriteForwardedExpr(Op.getState());
      OS << '.' << MemberNameOp.getName()
         << ";\n";
    } else {
      OS << "nbdl::get(";
      WriteForwardedExpr(Op.getState());
      if (!isa<nbdl_gen::UnitType>(Op.getKey().getType())) {
        OS << ", ";
        WriteForwardedExpr(Op.getKey());
      }
      OS << ");\n";
    }
  }

  void Visit(VisitOp Op) {
    WriteForwardedExpr(Op.getFn());
    OS << '(';
    llvm::interleave(Op.getArgs(), OS,
        [&](mlir::Value V) {
          WriteForwardedExpr(V);
        }, ",\n");
    OS << ");\n";
  }

  void Visit(MatchOp Op) {
    OS << "nbdl::match(";
    WriteForwardedExpr(Op.getStore());
    if (!isa<nbdl_gen::UnitType>(Op.getKey().getType())) {
      OS << ", ";
      WriteForwardedExpr(Op.getKey());
    }
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
    if (Body.empty())
      return;
    OS << "[&]";
    // Write parameters.
    OS << '(';
    mlir::BlockArgument& Arg = Body.getArguments().front();
    OS << Op.getType() << ' ' << SetLocalVarName(Arg, "arg_");
    OS << ')';
    OS << "{\n";
    VisitRegion(Body);
    OS << '}';
  }

  void Visit(MatchIfOp Op) {
    mlir::Region& Then = Op.getThenRegion();
    mlir::Region& Else = Op.getElseRegion();
    OS << "if (";
    WriteExpr(Op.getPred());
    OS << '(';
    WriteExpr(Op.getInput());
    OS << ")) {\n";
    VisitRegion(Then);

    // Check if the the else region is a single MatchIfOp
    // for pretty chaining.
    OS << "} else ";
    if (auto ChainedIfOp = dyn_cast<MatchIfOp>(Else.front().front())) {
      Visit(ChainedIfOp);
    } else {
      OS << "{\n";
      VisitRegion(Op.getElseRegion());
      OS << "}\n";
    }
  }

  void Visit(ApplyOp Op) {
    OS << "auto&& "
       << SetLocalVarName(Op.getResult(), "apply_")
       << " = ";
    // No forwarding stuff here
    WriteExpr(Op.getFn());
    OS << '(';
    llvm::interleaveComma(Op.getArgs(), OS,
        [&](mlir::Value Val) {
          WriteExpr(Val);
        });
    OS << ");\n";
  }

  void Visit(MemberNameOp Op) {
    // Member name is meaningless without the parent object
    // so we print it in GetOp.
    // We could implement in MatchOp, but it is a very
    // unlikely use case.
  }
};

class ContextWriter : public NbdlWriter<ContextWriter> {
public:
  // Record the values for each member in order.
  llvm::SmallVector<mlir::Value, 8> Members;

  using NbdlWriter<ContextWriter>::NbdlWriter;

  void VisitContext(ContextOp Op) {
    SetLoc(Op.getLoc());
    // Skip externally defined stores.
    if (Op.isExternal())
      return;

    ValueMapScope Scope(ValueMap);

    // Set the arg names first.
    for (mlir::BlockArgument BlockArg : Op.getBody().getArguments())
      SetLocalVarName(BlockArg, "arg_");

    // Delete both copy constructors to support subsumption with `auto&&`.
    OS << "class " << Op.getName() << " {\n";
    OS << "public:\n";
    WriteMemberDecls(Op);
    OS << Op.getName() << '(' << Op.getName() << " const&) = delete;\n";
    OS << Op.getName() << '(' << Op.getName() << "&) = delete;\n";
    WriteConstructor(Op);
    WriteAccessors();
    OS << "};\n";
    Flush();
  }

  nbdl_gen::ContOp getContOp(ContextOp Op) {
    mlir::Operation* Terminator = Op.getBody().front().getTerminator();
    auto ContOp = dyn_cast<nbdl_gen::ContOp>(Terminator);
    if (!ContOp)
      SetError("expecting nbdl.cont as terminator", Op);
    return ContOp;
  }

  void WriteMemberDecls(ContextOp Op) {
    // Get the ContOp and work backwards
    // saving the member names as we go.
    auto ContOp = getContOp(Op);
    if (!ContOp)
      return;

    WriteMemberDeclRec(ContOp.getArg());
  }

  void WriteMemberDeclRec(mlir::Value Val) {
    if (CheckError())
      return;

    if (auto SCO = Val.getDefiningOp<nbdl_gen::StoreComposeOp>()) {
      WriteMemberDeclRec(SCO);
    } else if (auto EO = Val.getDefiningOp<nbdl_gen::EmptyOp>();
               EO || isa<mlir::BlockArgument>(Val)) {
      return;
    } else {
      mlir::Operation* Irr = Val.getDefiningOp();
      SetError("unhandled operation: (WriteMemberDecl) {}", Irr);
    }
  }

  void WriteMemberDeclRec(StoreComposeOp Op) {
    WriteMemberDeclRec(Op.getRhs());

    heavy::SourceLocation PrevLoc = CurLoc;
    SetLoc(Op.getLoc());

    mlir::Value Key = Op.getKey();
    mlir::Value Lhs = Op.getLhs();
    llvm::StringRef Name;

    if (auto MemberNameOp = Key.getDefiningOp<nbdl_gen::MemberNameOp>()) {
      // It would be more consistent with our definition of StoreCompose
      // to support shadowing here, but since it is more work to check
      // and very suboptimal for the C++ compiler
      // and very likely to be a result of programming error,
      // we allow it to slip by and the C++ compiler will present the
      // user with an error.
      Name = SetLocalVal(Lhs, MemberNameOp.getName());
    } else {
      Name = SetLocalVarName(Lhs);
    }

    Members.push_back(Lhs);
    VisitType(Lhs);
    OS << ' ' << Name << ";\n";

    Flush();
    CurLoc = PrevLoc;
  }

  void WriteConstructor(ContextOp Op) {
    auto ContOp = getContOp(Op);
    if (!ContOp)
      return;

    OS << Op.getName();
    OS << '(';
    llvm::interleaveComma(Op.getBody().getArguments(), OS,
        [&](mlir::BlockArgument const& Arg) {
          OS << "auto&& " << GetLocalVal(Arg);
        });
    OS << ") \n: ";

    llvm::interleaveComma(Members, OS,
        [&](mlir::Value M) {
          OS << GetLocalVal(M);
          OS << '(';
          WriteInitArgs(M);
          OS << ')';
        });
    OS << "\n{ }\n";
  }

  void WriteInitArgs(mlir::Value Member) {
    if (auto StoreOp = Member.getDefiningOp<nbdl_gen::StoreOp>()) {
      llvm::interleaveComma(StoreOp.getArgs(), OS,
        [&](mlir::Value Arg) {
          WriteForwardedExpr(Arg);
        });
    } else {
      SetErrorV("unsupported operation arguments", Member);
    }
  }

  void WriteAccessors() {
    for (mlir::Value Value : Members) {
      OS << "decltype(auto) get_" << GetLocalVal(Value)
        << "() const {\n  return " << GetLocalVal(Value) << ";\n}\n";
    }
  }
};

}  // end namespace


namespace nbdl_gen {
std::tuple<std::string, heavy::SourceLocationEncoding*, mlir::Operation*>
translate_cpp(heavy::LexerWriterFnRef LexerWriter, mlir::Operation* Op) {
  if (auto FuncOp = dyn_cast<mlir::func::FuncOp>(Op)) {
    FuncWriter Writer(LexerWriter);
    Writer.Visit(Op);
    return std::make_tuple(std::move(Writer.ErrMsg),
                           Writer.ErrLoc, Writer.Irritant);
  } else if (auto ContextOp = dyn_cast<nbdl_gen::ContextOp>(Op)) {
    ContextWriter Writer(LexerWriter);
    Writer.VisitContext(ContextOp);
    return std::make_tuple(std::move(Writer.ErrMsg),
                           Writer.ErrLoc, Writer.Irritant);
  } else {
    return std::make_tuple(std::string("unhandled operation"),
                static_cast<heavy::SourceLocationEncoding*>(nullptr), Op);
  }

}
}
