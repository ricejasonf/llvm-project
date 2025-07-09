//===--- Nbdl.cpp - Nbdl binding syntax for HeavyScheme ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines syntax (nbdl impl) bindings for HeavyScheme.
//
//===----------------------------------------------------------------------===//

#include <nbdl_gen/Dialect.h>
#include <heavy/Context.h>
#include <heavy/MlirHelper.h>
#include <heavy/Nbdl.h>
#include <heavy/Value.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <optional>
#include <tuple>

namespace nbdl_gen {
std::tuple<std::string, heavy::SourceLocationEncoding*, mlir::Operation*>
translate_cpp(llvm::raw_ostream& OS, mlir::Operation* Op);
}

namespace heavy::nbdl_bind_var {
heavy::ContextLocal current_nbdl_module;
heavy::ExternFunction translate_cpp;
heavy::ExternFunction build_match_params_impl;
}

namespace {
// Create a builder that appends to the nbdl module.
std::optional<mlir::OpBuilder> getModuleBuilder(heavy::Context& C) {
  heavy::Value V = heavy::nbdl_bind_var::current_nbdl_module.get(C);
  mlir::Operation* Op = heavy::cast<mlir::Operation>(V);
  mlir::ModuleOp ModuleOp = heavy::dyn_cast_or_null<mlir::ModuleOp>(Op);
  if (!ModuleOp) {
    heavy::Error* E = C.CreateError(C.getLoc(),
        "invalid current-nbdl-module", heavy::Empty());
    C.Raise(E);
    return {};
  }
  mlir::OpBuilder Builder(ModuleOp);
  Builder.setInsertionPointToEnd(ModuleOp.getBody());
  return Builder;
}
}

namespace heavy::nbdl_bind {
// Create a function and call the thunk with a new builder
// to insert into the function body.
// _num_params_ does not include the store parameter.
// _callback_ takes _num_params_ + 1 arguments which are the block arguments.
// (%build_match_params _name_ _num_params_ _callback_)
void build_match_params_impl(Context& C, ValueRefs Args) {
  if (Args.size() != 3)
    return C.RaiseError("invalid arity");

  std::optional<mlir::OpBuilder> BuilderOpt = getModuleBuilder(C);
  if (!BuilderOpt)
    return;
  mlir::OpBuilder Builder = BuilderOpt.value();

  heavy::SourceLocation Loc = Args[0].getSourceLocation();
  // Require a heavy::Symbol so it has a source location.
  heavy::Symbol* NameSym = dyn_cast<heavy::Symbol>(Args[0]);
  llvm::StringRef Name = NameSym->getStringRef();
  heavy::Value NumParamsVal = Args[1];
  heavy::Value Callback = Args[2];

  int32_t NumParams = isa<heavy::Int>(NumParamsVal)
      ? int32_t(cast<heavy::Int>(NumParamsVal))
      : int32_t(-1);

  if (NumParams < 0)
    return C.RaiseError("expecting positive integer for num_params");
  if (!NameSym || Name.empty())
    return C.RaiseError("expecting function name (symbol literal)");

  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder.getContext());

  // Create the function type.
  llvm::SmallVector<mlir::Type, 8> InputTypes{
    Builder.getType<nbdl_gen::StoreType>()};
  for (unsigned i = 0; i < static_cast<uint32_t>(NumParams); i++)
    InputTypes.push_back(Builder.getType<nbdl_gen::OpaqueType>());
  mlir::FunctionType FT = Builder.getFunctionType(InputTypes,
                                                  /*ResultTypes*/{});

  // Create the function.
  auto FuncOp = Builder.create<mlir::func::FuncOp>(MLoc, Name, FT);
  FuncOp.addEntryBlock();

  heavy::Value Thunk = C.CreateLambda([FuncOp](Context& C, ValueRefs) mutable {
    heavy::Value Callback = C.getCapture(0);
    llvm::SmallVector<heavy::Value, 8> BlockArgs;
    assert(!FuncOp.getBody().empty() && "should have entry block");
    mlir::Block& EntryBlock = FuncOp.getBody().front();
    for (mlir::Value MVal : EntryBlock.getArguments()) {
      heavy::Value V = mlir_helper::createTagged(C,
          mlir_helper::kind::mlir_value, MVal);
      BlockArgs.push_back(V);
    }

    C.Apply(Callback, BlockArgs);
  }, CaptureList{Callback});

  // Call the thunk with a Builder at the entry point.
  Builder = mlir::OpBuilder(FuncOp.getBody());
  mlir_helper::with_builder_impl(C, Builder, Thunk);
}

// Translate a nbdl dialect operation to C++.
// (translate-cpp op port)
// Currently the "port" has to be a tagged llvm::raw_ostream.
void translate_cpp(Context& C, ValueRefs Args) {
  if (Args.size() != 2 && Args.size() != 1)
    return C.RaiseError("invalid arity");
  auto* Op = dyn_cast<mlir::Operation>(Args[0]);
  if (!Op)
    return C.RaiseError("expecting mlir.operation");

  llvm::raw_ostream* OS = nullptr;

  // Do not capture the emphemeral Tagged object.
  if (Args.size() == 2) {
    auto* Tagged = dyn_cast<heavy::Tagged>(Args[1]);
    heavy::Symbol* KindSym = C.CreateSymbol("::llvm::raw_ostream");
    if (!Tagged || !Tagged->isa(KindSym))
      return C.RaiseError("expecting ::llvm::raw_ostream");
    OS = &(Tagged->cast<llvm::raw_ostream>());
  } else {
    OS = &llvm::outs();
  }

  auto&& [ErrMsg, ErrLoc, Irritant] = nbdl_gen::translate_cpp(*OS, Op);
  if (!ErrMsg.empty()) {
    heavy::SourceLocation Loc(ErrLoc);
    heavy::Error* Err = C.CreateError(Loc, ErrMsg,
        Irritant ? heavy::Value(Irritant) : Undefined());
    return C.Raise(Err);
  }
  C.Cont();
}
}

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_NBDL_INIT(heavy::Context& C) {
  C.DialectRegistry->insert<nbdl_gen::NbdlDialect>();

  // Assume that the MLIRContext cleans up ModuleOps.
  mlir::OpBuilder Builder(C.MLIRContext.get());
  mlir::Location Loc = Builder.getUnknownLoc();
  mlir::ModuleOp ModuleOp
    = Builder.create<mlir::ModuleOp>(Loc, "nbdl_gen_module");

  heavy::nbdl_bind_var::current_nbdl_module.init(C, ModuleOp.getOperation());
  heavy::nbdl_bind_var::translate_cpp = heavy::nbdl_bind::translate_cpp;
  heavy::nbdl_bind_var::build_match_params_impl
    = heavy::nbdl_bind::build_match_params_impl;
}

void HEAVY_NBDL_LOAD_MODULE(heavy::Context& C) {
  HEAVY_NBDL_INIT(C);
  heavy::initModuleNames(C, HEAVY_NBDL_LIB_STR, {
    {"current-nbdl-module", heavy::nbdl_bind_var::current_nbdl_module.get(C)},
    {"translate-cpp", heavy::nbdl_bind_var::translate_cpp},
    {"%build-match-params", heavy::nbdl_bind_var::build_match_params_impl},
  });
}
}
