//===- Builtins.h - Builtin functions for HeavyScheme -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines HeavyScheme decalarations for values and evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_BUILTINS_H
#define LLVM_HEAVY_BUILTINS_H

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class Value;

}

namespace heavy {

class Context;
class Value;
class OpGen;
class OpEval;
class Pair;
using ValueRefs = llvm::ArrayRef<heavy::Value>;

}

namespace heavy { namespace builtin_syntax {

mlir::Value define(OpGen& OG, Pair* P);
mlir::Value if_(OpGen& OG, Pair* P);
mlir::Value lambda(OpGen& OG, Pair* P);
mlir::Value quasiquote(OpGen& C, Pair* P); // lib/Quasiquote.cpp
mlir::Value quote(OpGen& OG, Pair* P);     // lib/Quasiquote.cpp
mlir::Value set(OpGen& OG, Pair* P);

}}

namespace heavy { namespace builtin {

heavy::Value eval(Context& C, ValueRefs Args);
heavy::Value operator_add(Context& C, ValueRefs Args);
heavy::Value operator_mul(Context&C, ValueRefs Args);
heavy::Value operator_sub(Context&C, ValueRefs Args);
heavy::Value operator_div(Context& C, ValueRefs Args);
heavy::Value operator_gt(Context& C, ValueRefs Args);
heavy::Value operator_lt(Context& C, ValueRefs Args);
heavy::Value eq(Context& C, ValueRefs Args);
heavy::Value equal(Context& C, ValueRefs Args);
heavy::Value eqv(Context& C, ValueRefs Args);
heavy::Value list(Context& C, ValueRefs Args);
heavy::Value append(Context& C, ValueRefs Args);

}}

#endif

