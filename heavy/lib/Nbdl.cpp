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
translate_cpp(heavy::LexerWriterFnRef FnRef, mlir::Operation* Op);
}

namespace heavy::nbdl_bind_var {
heavy::ContextLocal current_nbdl_module;
heavy::ExternFunction translate_cpp;
heavy::ExternFunction close_previous_scope;
heavy::ExternFunction build_match_params_impl;
heavy::ExternFunction build_overload_impl;
heavy::ExternFunction build_match_if_impl;
heavy::ExternFunction build_context_impl;
heavy::ExternFunction build_match_op_impl;
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
// _num_store_params_ N
// _callback_ takes _num_store_params_ + 1 arguments which are the block arguments
//  with formals like (store1 store2 ... storeN fn)
// (%build_match_params _name_ _num_store_params_ _callback_)
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
    return C.RaiseError("expecting positive integer for num_store_params");
  if (!NameSym || Name.empty())
    return C.RaiseError("expecting function name (symbol literal)");

  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder.getContext());

  // Create the function type.
  llvm::SmallVector<mlir::Type, 8> InputTypes;
  for (unsigned i = 0; i < static_cast<uint32_t>(NumParams); i++)
    InputTypes.push_back(Builder.getType<nbdl_gen::StoreType>());

  // Push the visitor fn argument.
  InputTypes.push_back(Builder.getType<nbdl_gen::StoreType>());

  mlir::FunctionType FT = Builder.getFunctionType(InputTypes,
                                                  /*ResultTypes*/{});

  // Create the function.
  auto FuncOp = mlir::func::FuncOp::create(Builder, MLoc, Name, FT);
  FuncOp.addEntryBlock();

  heavy::Value Thunk = C.CreateLambda([FuncOp](Context& C, ValueRefs) mutable {
    heavy::Value Callback = C.getCapture(0);
    llvm::SmallVector<heavy::Value, 8> BlockArgs;
    assert(!FuncOp.getBody().empty() && "should have entry block");
    for (mlir::Value MVal : FuncOp.getBody().getArguments()) {
      heavy::Value V = C.CreateAny(MVal);
      BlockArgs.push_back(V);
    }

    C.PushCont([FuncOp](Context& C, ValueRefs) mutable {
      C.Cont(FuncOp.getOperation());
    }, CaptureList{});
    C.Apply(Callback, BlockArgs);
  }, CaptureList{Callback});

  // Call the thunk with a Builder at the entry point.
  Builder = mlir::OpBuilder(FuncOp.getBody());
  mlir_helper::with_builder_impl(C, Builder, Thunk);
}

// _callback_ takes a single block argument if specified.
// (%build_overload _loc_ _typename_ _callback_)
// (%build_overload _loc_ _typename_)
void build_overload_impl(Context& C, ValueRefs Args) {
  if (Args.size() != 2 && Args.size() != 3)
    return C.RaiseError("invalid arity");

  mlir::OpBuilder* Builder = mlir_helper::getCurrentBuilder(C);
  if (!Builder)
    return;

  // Create the operation with an entry block with a single argument.
  heavy::SourceLocation Loc = Args[0].getSourceLocation();
  heavy::Value TypenameArg = Args[1];
  heavy::Value Callback = Args.size() == 3 ? Args[2] : nullptr;

  if (!isa<String>(TypenameArg) && !isa<Symbol>(TypenameArg))
    return C.RaiseError("expecting string-like object for typename");

  llvm::StringRef Typename = TypenameArg.getStringRef();
  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder->getContext());
  auto OverloadOp = nbdl_gen::OverloadOp::create(*Builder, MLoc, Typename);
  if (!Callback)
    return C.Cont();

  OverloadOp.getBody().emplaceBlock();
  assert(OverloadOp.getBody().hasOneBlock() && "expecting a single block");

  heavy::Value Thunk = C.CreateLambda(
    [OverloadOp](Context& C, ValueRefs) mutable {
      heavy::Value Callback = C.getCapture(0);
      mlir::Region& Body = OverloadOp.getBody();
      mlir::Location MLoc = OverloadOp.getLoc();
      mlir::Type Type = mlir::OpBuilder(OverloadOp)
        .getType<nbdl_gen::StoreType>();
      mlir::BlockArgument BlockArg = Body.addArgument(Type, MLoc);
      heavy::Value V = C.CreateAny(mlir::Value(BlockArg));
      C.Apply(Callback, V);
    }, CaptureList{Callback});

  // Call the thunk with a Builder at the entry point.
  mlir::OpBuilder NewBuilder = mlir::OpBuilder(OverloadOp.getBody());
  mlir_helper::with_builder_impl(C, NewBuilder, Thunk);
}

void build_match_if_impl(Context& C, ValueRefs Args) {
  if (Args.size() != 5)
    return C.RaiseError("invalid arity");

  mlir::OpBuilder* Builder = mlir_helper::getCurrentBuilder(C);
  if (!Builder)
    return;

  // Create the operation with an entry block with a single argument.
  heavy::SourceLocation Loc = Args[0].getSourceLocation();
  mlir::Value Input = any_cast<mlir::Value>(Args[1]);
  mlir::Value Pred = any_cast<mlir::Value>(Args[2]);
  heavy::Value ThenThunk = Args[3];
  heavy::Value ElseThunk = Args[4];

  if (!Pred || !Input)
    return C.RaiseError("expecting mlir.value");

  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder->getContext());
  auto MatchIfOp = nbdl_gen::MatchIfOp::create(*Builder, MLoc, Input, Pred);
  MatchIfOp.getThenRegion().emplaceBlock();
  MatchIfOp.getElseRegion().emplaceBlock();

  C.PushCont([MatchIfOp](Context& C, ValueRefs Args) mutable {
    heavy::Value ElseThunk = C.getCapture(0);
    mlir::OpBuilder ElseBuilder(MatchIfOp.getElseRegion());
    mlir_helper::with_builder_impl(C, ElseBuilder, ElseThunk);
  }, CaptureList{ElseThunk});

  mlir::OpBuilder ThenBuilder(MatchIfOp.getThenRegion());
  mlir_helper::with_builder_impl(C, ThenBuilder, ThenThunk);
}

// Translate a nbdl dialect operation to C++.
// (translate-cpp op port)
// The parameter `op` may be an mlir::Operation* or a StringLike
// which will be used to look up the name in the module.
// Currently the "port" has to be a tagged llvm::raw_ostream.
void translate_cpp(Context& C, ValueRefs Args) {
  if (Args.size() != 2 && Args.size() != 1)
    return C.RaiseError("invalid arity");
  auto* Op = dyn_cast<mlir::Operation>(Args[0]);
  if (!Op)
    return C.RaiseError("expecting mlir.operation: {}", Args[0]);

  llvm::raw_ostream* OS = nullptr;

  using ResultTy = std::tuple<std::string,
                              heavy::SourceLocationEncoding*,
                              mlir::Operation*>;
  auto Result = ResultTy();


  // Do not capture the emphemeral Any object.
  if (Args.size() == 2) {
    if (auto LWF = any_cast<LexerWriterFnRef>(Args[1])) {
      Result = nbdl_gen::translate_cpp(LWF, Op);
    } else if (auto* Raw = any_cast<::llvm::raw_ostream>(&Args[1])) {
      OS = Raw;
    } else {
      return C.RaiseError("expecting llvm::raw_ostream"
                          " or heavy::LexerWriterFnRef");
    }
  } else {
    OS = &llvm::outs();
  }
  if (OS) {
    auto LexerWriter = [&OS](heavy::SourceLocation, llvm::StringRef Buffer) {
      *OS << Buffer;
    };
    Result = nbdl_gen::translate_cpp(LexerWriter, Op);
  }

  auto& [ErrMsg, ErrLoc, Irritant] = Result;
  if (!ErrMsg.empty()) {
    heavy::SourceLocation Loc(ErrLoc);
    heavy::Error* Err = C.CreateError(Loc, ErrMsg,
        Irritant ? heavy::Value(Irritant) : Value(Undefined()));
    return C.Raise(Err);
  }
  C.Cont();
}

// If the current block has a terminator, wrap the
// entire block in a nbdl.scope. This supports the
// convention that only terminators may perform an
// operation that may invalidate child stores.
void close_previous_scope(Context& C, ValueRefs Args) {
  if (Args.size() != 0)
    return C.RaiseError("invalid arity");
  mlir::OpBuilder* Builder = mlir_helper::getCurrentBuilder(C);
  if (!Builder)
    return;  // error is already raised by getCurrentBuilder
  mlir::Block* Block = Builder->getBlock();
  assert(Block && isa<mlir::func::FuncOp>(Block->getParentOp())
      && "expecting func insertion point");
  if (Block->empty() || !Block->back().hasTrait<mlir::OpTrait::IsTerminator>())
    return C.Cont();

  mlir::Location Loc = Block->back().getLoc();

  // Create new Region for ScopeOp.
  auto ScopeBody = std::make_unique<mlir::Region>();
  mlir::Block& NewBlock = ScopeBody->emplaceBlock();
  while (!Block->empty())
    Block->front().moveBefore(&NewBlock, NewBlock.end());
  mlir::Operation* ScopeOp = nbdl_gen::ScopeOp::create(
                              *Builder, Loc, std::move(ScopeBody));
  Builder->setInsertionPointAfter(ScopeOp);

  C.Cont();
}

void build_context_impl(Context& C, ValueRefs Args) {
  return C.RaiseError("deprecated");
#if 0
  if (Args.size() != 3)
    return C.RaiseError("invalid arity");

  std::optional<mlir::OpBuilder> BuilderOpt = getModuleBuilder(C);
  if (!BuilderOpt)
    return;
  mlir::OpBuilder Builder = BuilderOpt.value();

  // This part is exactly like build_match_params_impl
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
  // Create the ContextOp.
  auto ContextOp = Builder.create<nbdl_gen::ContextOp>(MLoc, Name);
  mlir::Block& EntryBlock = ContextOp.getBody().emplaceBlock();

  // Create the arguments.
  auto StoreTy = Builder.getType<nbdl_gen::StoreType>();
  for (int32_t i = 0; i < NumParams; i++)
    EntryBlock.addArgument(StoreTy, MLoc);

  heavy::Value Thunk = C.CreateLambda([ContextOp](Context& C,
                                                  ValueRefs) mutable {
    heavy::Value Callback = C.getCapture(0);
    llvm::SmallVector<heavy::Value, 8> BlockArgs;
    assert(!ContextOp.getBody().empty() && "should have entry block");
    for (mlir::Value MVal : ContextOp.getBody().getArguments()) {
      heavy::Value V = C.CreateAny(MVal);
      BlockArgs.push_back(V);
    }

    C.Apply(Callback, BlockArgs);
  }, CaptureList{Callback});

  // Call the thunk with a Builder at the entry point.
  Builder = mlir::OpBuilder(ContextOp.getBody());
  mlir_helper::with_builder_impl(C, Builder, Thunk);
#endif
}


// (%build-match-op name storeval keyval
void build_match_op_impl(Context& C, ValueRefs Args) {
  C.Cont();
}
} //  end namespace heavy::nbdl_bind

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_NBDL_INIT(heavy::Context& C) {
  C.DialectRegistry->insert<nbdl_gen::NbdlDialect>();

  // Assume that the MLIRContext cleans up ModuleOps.
  mlir::OpBuilder Builder(C.MLIRContext.get());
  mlir::Location Loc = Builder.getUnknownLoc();
  mlir::ModuleOp ModuleOp
    = mlir::ModuleOp::create(Builder, Loc, "nbdl_gen_module");

  heavy::nbdl_bind_var::current_nbdl_module.set(C, ModuleOp.getOperation());
  heavy::nbdl_bind_var::translate_cpp = heavy::nbdl_bind::translate_cpp;
  heavy::nbdl_bind_var::close_previous_scope
    = heavy::nbdl_bind::close_previous_scope;
  heavy::nbdl_bind_var::build_match_params_impl
    = heavy::nbdl_bind::build_match_params_impl;
  heavy::nbdl_bind_var::build_overload_impl
    = heavy::nbdl_bind::build_overload_impl;
  heavy::nbdl_bind_var::build_match_if_impl
    = heavy::nbdl_bind::build_match_if_impl;
  heavy::nbdl_bind_var::build_context_impl
    = heavy::nbdl_bind::build_context_impl;
  heavy::nbdl_bind_var::build_match_op_impl
    = heavy::nbdl_bind::build_match_op_impl;
}

void HEAVY_NBDL_LOAD_MODULE(heavy::Context& C) {
  HEAVY_NBDL_INIT(C);
  heavy::initModuleNames(C, HEAVY_NBDL_LIB_STR, {
    {"current-nbdl-module", heavy::nbdl_bind_var::current_nbdl_module.get(C)},
    {"translate-cpp", heavy::nbdl_bind_var::translate_cpp},
    {"close-previous-scope",
                  heavy::nbdl_bind_var::close_previous_scope},
    {"%build-match-params", heavy::nbdl_bind_var::build_match_params_impl},
    {"%build-overload", heavy::nbdl_bind_var::build_overload_impl},
    {"%build-match-if", heavy::nbdl_bind_var::build_match_if_impl},
    {"%build-context", heavy::nbdl_bind_var::build_context_impl},
    {"%build-match-op", heavy::nbdl_bind_var::build_match_op_impl},
  });
}
}
