//===- Base.h - Base library functions for HeavyScheme ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares values and functions for the (heavy base) scheme library
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_BUILTINS_H
#define LLVM_HEAVY_BUILTINS_H

#include "heavy/Value.h"

#define HEAVY_BASE_LIB                _HEAVYL5SheavyL4Sbase
#define HEAVY_BASE_LIB_(NAME)         _HEAVYL5SheavyL4Sbase ## NAME
#define HEAVY_BASE_LIB_STR            "_HEAVYL5SheavyL4Sbase"
#define HEAVY_BASE_IS_LOADED          HEAVY_BASE_LIB_(_is_loaded)
#define HEAVY_BASE_LOAD_MODULE        HEAVY_BASE_LIB_(_load_module)
#define HEAVY_BASE_INIT               HEAVY_BASE_LIB_(_init)
#define HEAVY_BASE_VAR(NAME)          HEAVY_BASE_VAR__##NAME
#define HEAVY_BASE_VAR__import        HEAVY_BASE_LIB_(V6Simport)
#define HEAVY_BASE_VAR__add           HEAVY_BASE_LIB_(Vpl)
#define HEAVY_BASE_VAR__sub           HEAVY_BASE_LIB_(Vmi)
#define HEAVY_BASE_VAR__div           HEAVY_BASE_LIB_(Vdv)
#define HEAVY_BASE_VAR__mul           HEAVY_BASE_LIB_(Vml)
#define HEAVY_BASE_VAR__gt            HEAVY_BASE_LIB_(Vgt)
#define HEAVY_BASE_VAR__lt            HEAVY_BASE_LIB_(Vlt)
#define HEAVY_BASE_VAR__list          HEAVY_BASE_LIB_(V4Slist)
#define HEAVY_BASE_VAR__append        HEAVY_BASE_LIB_(V6Sappend)
#define HEAVY_BASE_VAR__dump          HEAVY_BASE_LIB_(V4Sdump)
#define HEAVY_BASE_VAR__eq            HEAVY_BASE_LIB_(V2Seqqu)
#define HEAVY_BASE_VAR__equal         HEAVY_BASE_LIB_(V5Sequalqu)
#define HEAVY_BASE_VAR__eqv           HEAVY_BASE_LIB_(V3Seqvqu)
#define HEAVY_BASE_VAR__eval          HEAVY_BASE_LIB_(V4Seval)
#define HEAVY_BASE_VAR__callcc        HEAVY_BASE_LIB_(V4Scalldv2Scc)

namespace mlir {

class Value;

}

namespace heavy {

class Context;
class Value;
class OpGen;
class OpEval;
class Pair;
using ValueRefs = llvm::MutableArrayRef<heavy::Value>;

}

namespace heavy { namespace base {

// syntax
mlir::Value define(OpGen& OG, Pair* P);
mlir::Value if_(OpGen& OG, Pair* P);
mlir::Value lambda(OpGen& OG, Pair* P);
mlir::Value quasiquote(OpGen& C, Pair* P); // lib/Quasiquote.cpp
mlir::Value quote(OpGen& OG, Pair* P);     // lib/Quasiquote.cpp
mlir::Value set(OpGen& OG, Pair* P);
mlir::Value import(OpGen& OG, Pair* P);


// functions
heavy::Value eval(Context& C, ValueRefs Args);
heavy::Value dump(Context& C, ValueRefs Args);
heavy::Value add(Context& C, ValueRefs Args);
heavy::Value mul(Context&C, ValueRefs Args);
heavy::Value sub(Context&C, ValueRefs Args);
heavy::Value div(Context& C, ValueRefs Args);
heavy::Value gt(Context& C, ValueRefs Args);
heavy::Value lt(Context& C, ValueRefs Args);
heavy::Value eq(Context& C, ValueRefs Args);
heavy::Value equal(Context& C, ValueRefs Args);
heavy::Value eqv(Context& C, ValueRefs Args);
heavy::Value list(Context& C, ValueRefs Args);
heavy::Value append(Context& C, ValueRefs Args);
heavy::Value callcc(Context& C, ValueRefs Args);
heavy::Value with_exception_handler(Context& C, ValueRefs Args);
heavy::Value raise(Context& C, ValueRefs Args);
heavy::Value error(Context& C, ValueRefs Args);

}}

extern heavy::ExternSyntax   HEAVY_BASE_VAR(define);
extern heavy::ExternSyntax   HEAVY_BASE_VAR(if);
extern heavy::ExternSyntax   HEAVY_BASE_VAR(lambda);
extern heavy::ExternSyntax   HEAVY_BASE_VAR(quasiquote);
extern heavy::ExternSyntax   HEAVY_BASE_VAR(quote);
extern heavy::ExternSyntax   HEAVY_BASE_VAR(set);

extern heavy::ExternFunction HEAVY_BASE_VAR(add);
extern heavy::ExternFunction HEAVY_BASE_VAR(sub);
extern heavy::ExternFunction HEAVY_BASE_VAR(div);
extern heavy::ExternFunction HEAVY_BASE_VAR(mul);
extern heavy::ExternFunction HEAVY_BASE_VAR(gt);
extern heavy::ExternFunction HEAVY_BASE_VAR(lt);
extern heavy::ExternFunction HEAVY_BASE_VAR(list);
extern heavy::ExternFunction HEAVY_BASE_VAR(append);
extern heavy::ExternFunction HEAVY_BASE_VAR(dump);
extern heavy::ExternFunction HEAVY_BASE_VAR(eq);
extern heavy::ExternFunction HEAVY_BASE_VAR(equal);
extern heavy::ExternFunction HEAVY_BASE_VAR(eqv);
extern heavy::ExternFunction HEAVY_BASE_VAR(eval);
extern heavy::ExternFunction HEAVY_BASE_VAR(callcc);
extern heavy::ExternFunction HEAVY_BASE_VAR(with_exception_handler);
extern heavy::ExternFunction HEAVY_BASE_VAR(raise);
extern heavy::ExternFunction HEAVY_BASE_VAR(error);

extern bool HEAVY_BASE_IS_LOADED;

extern "C" {
// initialize the module for run-time independent of the compiler
inline void HEAVY_BASE_INIT(heavy::Context& Context) {
  assert(!HEAVY_BASE_IS_LOADED &&
    "module should not be loaded more than once");
  HEAVY_BASE_IS_LOADED = true;

  // syntax
  HEAVY_BASE_VAR(define)      = heavy::base::define;
  HEAVY_BASE_VAR(if)          = heavy::base::if_;
  HEAVY_BASE_VAR(lambda)      = heavy::base::lambda;
  HEAVY_BASE_VAR(quasiquote)  = heavy::base::quasiquote;
  HEAVY_BASE_VAR(quote)       = heavy::base::quote;
  HEAVY_BASE_VAR(set)         = heavy::base::set;

  // functions
  HEAVY_BASE_VAR(add)     = heavy::base::add;
  HEAVY_BASE_VAR(sub)     = heavy::base::sub;
  HEAVY_BASE_VAR(div)     = heavy::base::div;
  HEAVY_BASE_VAR(mul)     = heavy::base::mul;
  HEAVY_BASE_VAR(gt)      = heavy::base::gt;
  HEAVY_BASE_VAR(lt)      = heavy::base::lt;
  HEAVY_BASE_VAR(list)    = heavy::base::list;
  HEAVY_BASE_VAR(append)  = heavy::base::append;
  HEAVY_BASE_VAR(dump)    = heavy::base::dump;
  HEAVY_BASE_VAR(eq)      = heavy::base::eqv;
  HEAVY_BASE_VAR(equal)   = heavy::base::equal;
  HEAVY_BASE_VAR(eqv)     = heavy::base::eqv;
  HEAVY_BASE_VAR(eval)    = heavy::base::eval;
  HEAVY_BASE_VAR(callcc)  = heavy::base::callcc;
  HEAVY_BASE_VAR(with_exception_handler)
    = heavy::base::with_exception_handler;
  HEAVY_BASE_VAR(raise)   = heavy::base::raise;
  HEAVY_BASE_VAR(error)   = heavy::base::error;
}

// initializes the module and loads lookup information
// for the compiler
inline void HEAVY_BASE_LOAD_MODULE(heavy::Context& Context) {
  HEAVY_BASE_INIT(Context);
  heavy::initModule(Context, HEAVY_BASE_LIB_STR, {
    // syntax
    {"define",      HEAVY_BASE_VAR(define)},
    {"if",          HEAVY_BASE_VAR(if)},
    {"lambda",      HEAVY_BASE_VAR(lambda)},
    {"quasiquote",  HEAVY_BASE_VAR(quasiquote)},
    {"quote",       HEAVY_BASE_VAR(quote)},
    {"set!",        HEAVY_BASE_VAR(set)},

    // functions
    {"+",       HEAVY_BASE_VAR(add)},
    {"-",       HEAVY_BASE_VAR(sub)},
    {"/",       HEAVY_BASE_VAR(div)},
    {"*",       HEAVY_BASE_VAR(mul)},
    {">",       HEAVY_BASE_VAR(gt)},
    {"<",       HEAVY_BASE_VAR(lt)},
    {"list",    HEAVY_BASE_VAR(list)},
    {"append",  HEAVY_BASE_VAR(append)},
    {"dump",    HEAVY_BASE_VAR(dump)},
    {"eq?",     HEAVY_BASE_VAR(eq)},
    {"equal?",  HEAVY_BASE_VAR(equal)},
    {"eqv?",    HEAVY_BASE_VAR(eqv)},
    {"eval",    HEAVY_BASE_VAR(eval)},
    {"call/cc", HEAVY_BASE_VAR(callcc)},
    {"with-exception-handler", HEAVY_BASE_VAR(with_exception_handler)},
    {"raise", HEAVY_BASE_VAR(raise)},
    {"error", HEAVY_BASE_VAR(error)},
  });
}
}


#endif

