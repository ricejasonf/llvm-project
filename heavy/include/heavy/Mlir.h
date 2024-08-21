//===- Mlir.h - Mlir binding functions for HeavyScheme ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares values and functions for the (heavy mlir) scheme library
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_MLIR_H
#define LLVM_HEAVY_MLIR_H

#include "heavy/Value.h"

#define HEAVY_MLIR_LIB                _HEAVYL5SheavyL4Smlir
#define HEAVY_MLIR_LIB_(NAME)         _HEAVYL5SheavyL4Smlir ## NAME
#define HEAVY_MLIR_LIB_STR            "_HEAVYL5SheavyL4Smlir"
#define HEAVY_MLIR_LOAD_MODULE        HEAVY_MLIR_LIB_(_load_module)
#define HEAVY_MLIR_INIT               HEAVY_MLIR_LIB_(_init)

#define HEAVY_MLIR_VAR(NAME)          HEAVY_MLIR_VAR__##NAME
#define HEAVY_MLIR_VAR__create_op     HEAVY_MLIR_LIB_(V6Screatemi2Sop)
#define HEAVY_MLIR_VAR__create_op_impl \
                                      HEAVY_MLIR_LIB_(Vrm6Screatemi2Sop)
#define HEAVY_MLIR_VAR__region        HEAVY_MLIR_LIB_(V6Sregion)
#define HEAVY_MLIR_VAR__results       HEAVY_MLIR_LIB_(V7Sresults)
#define HEAVY_MLIR_VAR__result        HEAVY_MLIR_LIB_(V6Sresult)
#define HEAVY_MLIR_VAR__at_block_begin \
                                      HEAVY_MLIR_LIB_(V5Sblockmi5Sbegin)
#define HEAVY_MLIR_VAR__at_block_end  HEAVY_MLIR_LIB_(V5SSblockmi3end)
#define HEAVY_MLIR_VAR__block_arg     HEAVY_MLIR_LIB_(V5Sblockmi3Sarg)
#define HEAVY_MLIR_VAR__block_op      HEAVY_MLIR_LIB_(V5Sblockmi2Sop)
#define HEAVY_MLIR_VAR__op_next       HEAVY_MLIR_LIB_(V2Sopmi4Snext)
#define HEAVY_MLIR_VAR__parent_op     HEAVY_MLIR_LIB_(V6Sparentmi2Sop)
#define HEAVY_MLIR_VAR__set_insertion_point \
                                      HEAVY_MLIR_LIB_(V6Sinsertmi6Sbefore)
#define HEAVY_MLIR_VAR__set_insertion_after \
                                      HEAVY_MLIR_LIB_(V12Sinsert_after)
#define HEAVY_MLIR_VAR__type          HEAVY_MLIR_LIB_(V4Stype)
#define HEAVY_MLIR_VAR__attr          HEAVY_MLIR_LIB_(V4Sattr)

#define HEAVY_MLIR_VAR__current_context \
                                HEAVY_MLIR_LIB_(V7ScurrentmiV7Scontext)
#define HEAVY_MLIR_VAR__current_builder \
                                HEAVY_MLIR_LIB_(V7ScurrentmiV7Sbuilder)
#define HEAVY_MLIR_VAR__with_new_context \
                                HEAVY_MLIR_LIB_(V4SwithmiV3Snewmiv7Scontext)
#define HEAVY_MLIR_VAR__with_builder \
                                HEAVY_MLIR_LIB_(V4Swithmiv7Sbuilder)
#define HEAVY_MLIR_VAR__load_dialect \
                                HEAVY_MLIR_LIB_(V4SloadmiV7Sdialect)
#define HEAVY_MLIR_VAR__verify        HEAVY_MLIR_LIB_(V6Sverify)

#define HEAVY_MLIR_VAR_STR(X) HEAVY_MLIR_VAR_STR_STR(X)
#define HEAVY_MLIR_VAR_STR_STR(X) #X

namespace heavy {

class Context;
class Value;
class OpGen;
class OpEval;
class Pair;
using ValueRefs = llvm::MutableArrayRef<heavy::Value>;

}

namespace heavy::mlir_bind {
void create_op(Context& C, ValueRefs Args);
void create_op_impl(Context& C, ValueRefs Args); // %create-op
void region(Context& C, ValueRefs Args);
void region_blocks(Context& C, ValueRefs Args);
void results(Context& C, ValueRefs Args);
void result(Context& C, ValueRefs Args);
void at_block_begin(Context& C, ValueRefs Args);
void at_block_end(Context& C, ValueRefs Args);
void block_arg(Context& C, ValueRefs Args);
void block_op(Context& C, ValueRefs Args);
void op_next(Context& C, ValueRefs Args);
void parent_op(Context& C, ValueRefs Args);
void type(Context& C, ValueRefs Args);
void attr(Context& C, ValueRefs Args);
void set_insertion_point(Context& C, ValueRefs Args);
void set_insertion_after(Context& C, ValueRefs Args);
void with_new_context(Context& C, ValueRefs Args);
void with_builder(Context& C, ValueRefs Args);
void load_dialect(Context& C, ValueRefs Args);
void verify(Context& C, ValueRefs Args);
}

extern heavy::ContextLocal   HEAVY_MLIR_VAR(current_context);
extern heavy::ContextLocal   HEAVY_MLIR_VAR(current_builder);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(create_op);
extern heavy::ExternFunction<> HEAVY_MLIR_VAR(create_op_impl);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(region);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(region_blocks);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(results);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(result);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_begin);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_end);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(block_arg);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(block_op);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(op_next);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(parent_op);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(type);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(attr);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(set_insertion_point);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(set_insertion_after);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_new_context);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_builder);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(load_dialect);
extern heavy::ExternSyntax<> HEAVY_MLIR_VAR(verify);

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_MLIR_INIT(heavy::Context& Context);
void HEAVY_MLIR_LOAD_MODULE(heavy::Context& Context);
}

#endif  // LLVM_HEAVY_MLIR_H
