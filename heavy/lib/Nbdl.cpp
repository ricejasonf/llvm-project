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

#include <heavy/Context.h>
#include <heavy/Nbdl.h>
#include <heavy/Value.h>
#include <nbdl_gen/Dialect.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <tuple>

namespace nbdl_gen {
std::tuple<std::string, mlir::Location, mlir::Operation*>
translate_cpp(llvm::raw_ostream& OS, mlir::Operation* Op);
}

namespace heavy::nbdl_bind_var {
heavy::ExternFunction write_cpp;
}

namespace heavy::nbdl_bind {
// Translate a nbdl dialect operation to C++.
void write_cpp(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  auto* Op = dyn_cast<mlir::Operation>(Args[0]);
  // Do not captured the emphemeral Tagged object.
  auto* Tagged = dyn_cast<heavy::Tagged>(Args[1]);
  heavy::Symbol* KindSym = C.CreateSymbol("::llvm::raw_ostream");
  if (!Op)
    return C.RaiseError("expecting mlir.operation");
  if (!Tagged || !Tagged->isa(KindSym))
    return C.RaiseError("expecting ::llvm::raw_ostream");

  auto& OS = Tagged->cast<llvm::raw_ostream>();
  auto&& [ErrMsg, ErrLoc, Irritant] = nbdl_gen::translate_cpp(OS, Op);
  if (!ErrMsg.empty()) {
    auto Loc = heavy::SourceLocation(mlir::OpaqueLoc
      ::getUnderlyingLocationOrNull<heavy::SourceLocationEncoding*>(
        mlir::dyn_cast<mlir::OpaqueLoc>(ErrLoc)));
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

  heavy::nbdl_bind_var::write_cpp = heavy::nbdl_bind::write_cpp;
}

void HEAVY_NBDL_LOAD_MODULE(heavy::Context& C) {
  HEAVY_NBDL_INIT(C);
  heavy::initModuleNames(C, HEAVY_NBDL_LIB_STR, {
    {"write-cpp", heavy::nbdl_bind_var::write_cpp},
  });
}
}
