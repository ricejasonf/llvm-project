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

#include "heavy/Context.h"
#include "heavy/Value.h"
#include "llvm/ADT/SmallVector.h"

#define HEAVY_BASE_LIB                _HEAVYL5SheavyL4Sbase
#define HEAVY_BASE_LIB_(NAME)         _HEAVYL5SheavyL4Sbase ## NAME
#define HEAVY_BASE_LIB_STR            "_HEAVYL5SheavyL4Sbase"
#define HEAVY_BASE_LOAD_MODULE        HEAVY_BASE_LIB_(_load_module)
#define HEAVY_BASE_INIT               HEAVY_BASE_LIB_(_init)

#define HEAVY_BASE_VAR(NAME)          HEAVY_BASE_VAR__##NAME
#define HEAVY_BASE_VAR__export        HEAVY_BASE_LIB_(V6Sexport)
#define HEAVY_BASE_VAR__add           HEAVY_BASE_LIB_(Vpl)
#define HEAVY_BASE_VAR__sub           HEAVY_BASE_LIB_(Vmi)
#define HEAVY_BASE_VAR__div           HEAVY_BASE_LIB_(Vdv)
#define HEAVY_BASE_VAR__mul           HEAVY_BASE_LIB_(Vml)
#define HEAVY_BASE_VAR__gt            HEAVY_BASE_LIB_(Vgt)
#define HEAVY_BASE_VAR__lt            HEAVY_BASE_LIB_(Vlt)
#define HEAVY_BASE_VAR__list          HEAVY_BASE_LIB_(V4Slist)
#define HEAVY_BASE_VAR__append        HEAVY_BASE_LIB_(V6Sappend)
#define HEAVY_BASE_VAR__dump          HEAVY_BASE_LIB_(V4Sdump)
#define HEAVY_BASE_VAR__write         HEAVY_BASE_LIB_(V5Swrite)
#define HEAVY_BASE_VAR__newline       HEAVY_BASE_LIB_(V7Snewline)
#define HEAVY_BASE_VAR__eq            HEAVY_BASE_LIB_(V2Seqqu)
#define HEAVY_BASE_VAR__equal         HEAVY_BASE_LIB_(V5Sequalqu)
#define HEAVY_BASE_VAR__eqv           HEAVY_BASE_LIB_(V3Seqvqu)
#define HEAVY_BASE_VAR__eval          HEAVY_BASE_LIB_(V4Seval)
#define HEAVY_BASE_VAR__call_cc       HEAVY_BASE_LIB_(V4Scalldv2Scc)
#define HEAVY_BASE_VAR__include       HEAVY_BASE_LIB_(V7Sinclude)
#define HEAVY_BASE_VAR__parse_source_file \
                                      HEAVY_BASE_LIB_(V5Sparsemi6Ssourcemi4Sfile)
#define HEAVY_BASE_VAR__dynamic_wind  HEAVY_BASE_LIB_(V7Sdynamicmi4Swind)
#define HEAVY_BASE_VAR__module_path   HEAVY_BASE_LIB_(V6Smodulemi4Spath)

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

// syntax (top level, continuable)
void begin(Context& C, ValueRefs Args);
void define_library(Context& C, ValueRefs Args);
void export_(Context&, ValueRefs);
void import_(Context&, ValueRefs);
void include_(Context&, ValueRefs);
void include_ci(Context&, ValueRefs);
void include_library_declarations(Context&, ValueRefs);
void parse_source_file(Context&, ValueRefs);

// syntax
mlir::Value define(OpGen& OG, Pair* P);
mlir::Value define_syntax(OpGen& OG, Pair* P);
mlir::Value syntax_rules(OpGen& OG, Pair* P);
mlir::Value if_(OpGen& OG, Pair* P);
mlir::Value lambda(OpGen& OG, Pair* P);
mlir::Value quasiquote(OpGen& C, Pair* P); // lib/Quasiquote.cpp
mlir::Value quote(OpGen& OG, Pair* P);     // lib/Quasiquote.cpp
mlir::Value set(OpGen& OG, Pair* P);
mlir::Value cond_expand(OpGen& OG, Pair* P);
mlir::Value source_loc(OpGen& OG, Pair* P);


// functions
void dump(Context& C, ValueRefs Args);
void write(Context& C, ValueRefs Args);
void newline(Context& C, ValueRefs Args);
void add(Context& C, ValueRefs Args);
void mul(Context&C, ValueRefs Args);
void sub(Context&C, ValueRefs Args);
void div(Context& C, ValueRefs Args);
void gt(Context& C, ValueRefs Args);
void lt(Context& C, ValueRefs Args);
void eq(Context& C, ValueRefs Args);
void equal(Context& C, ValueRefs Args);
void eqv(Context& C, ValueRefs Args);
void list(Context& C, ValueRefs Args);
void append(Context& C, ValueRefs Args);
void call_cc(Context& C, ValueRefs Args);
void with_exception_handler(Context& C, ValueRefs Args);
void raise(Context& C, ValueRefs Args);
void error(Context& C, ValueRefs Args);
void dynamic_wind(Context& C, ValueRefs Args);

// TODO These should go in (heavy eval)
void eval(Context& C, ValueRefs Args);
void op_eval(Context& C, ValueRefs Args);
void compile(Context& C, ValueRefs Args);

}}


extern heavy::ExternSyntax<>        HEAVY_BASE_VAR(begin);
extern heavy::ExternSyntax<>        HEAVY_BASE_VAR(define_library);
extern heavy::ExternSyntax<>        HEAVY_BASE_VAR(export);
extern heavy::ExternSyntax<>        HEAVY_BASE_VAR(include);
extern heavy::ExternSyntax<>        HEAVY_BASE_VAR(include_ci);
extern heavy::ExternSyntax<>     
  HEAVY_BASE_VAR(include_library_declarations);
extern heavy::ContextLocal          HEAVY_BASE_VAR(parse_source_file);

extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(define);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(define_syntax);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(syntax_rules);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(if);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(lambda);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(quasiquote);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(quote);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(set);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(cond_expand);
extern heavy::ExternBuiltinSyntax   HEAVY_BASE_VAR(source_loc);

extern heavy::ExternFunction HEAVY_BASE_VAR(add);
extern heavy::ExternFunction HEAVY_BASE_VAR(sub);
extern heavy::ExternFunction HEAVY_BASE_VAR(div);
extern heavy::ExternFunction HEAVY_BASE_VAR(mul);
extern heavy::ExternFunction HEAVY_BASE_VAR(gt);
extern heavy::ExternFunction HEAVY_BASE_VAR(lt);
extern heavy::ExternFunction HEAVY_BASE_VAR(list);
extern heavy::ExternFunction HEAVY_BASE_VAR(append);
extern heavy::ExternFunction HEAVY_BASE_VAR(dump);
extern heavy::ExternFunction HEAVY_BASE_VAR(write);
extern heavy::ExternFunction HEAVY_BASE_VAR(newline);
extern heavy::ExternFunction HEAVY_BASE_VAR(eq);
extern heavy::ExternFunction HEAVY_BASE_VAR(equal);
extern heavy::ExternFunction HEAVY_BASE_VAR(eqv);
extern heavy::ExternFunction HEAVY_BASE_VAR(call_cc);
extern heavy::ExternFunction HEAVY_BASE_VAR(with_exception_handler);
extern heavy::ExternFunction HEAVY_BASE_VAR(raise);
extern heavy::ExternFunction HEAVY_BASE_VAR(error);
extern heavy::ExternFunction HEAVY_BASE_VAR(dynamic_wind);

// TODO Put this stuff in a "eval" submodule or something.
extern heavy::ExternFunction HEAVY_BASE_VAR(eval);
extern heavy::ExternFunction HEAVY_BASE_VAR(op_eval);
extern heavy::ExternFunction HEAVY_BASE_VAR(compile);
extern heavy::ContextLocal   HEAVY_BASE_VAR(module_path);

extern "C" {
// initialize the module for run-time independent of the compiler
inline void HEAVY_BASE_INIT(heavy::Context& Context) {
  // syntax
  HEAVY_BASE_VAR(define)          = heavy::base::define;
  HEAVY_BASE_VAR(define_syntax)   = heavy::base::define_syntax;
  HEAVY_BASE_VAR(syntax_rules)    = heavy::base::syntax_rules;
  HEAVY_BASE_VAR(if)              = heavy::base::if_;
  HEAVY_BASE_VAR(lambda)          = heavy::base::lambda;
  HEAVY_BASE_VAR(quasiquote)      = heavy::base::quasiquote;
  HEAVY_BASE_VAR(quote)           = heavy::base::quote;
  HEAVY_BASE_VAR(set)             = heavy::base::set;
  HEAVY_BASE_VAR(begin)           = heavy::base::begin;
  HEAVY_BASE_VAR(cond_expand)     = heavy::base::cond_expand;
  HEAVY_BASE_VAR(define_library)  = heavy::base::define_library;
  HEAVY_BASE_VAR(export)          = heavy::base::export_;
  HEAVY_BASE_VAR(include)         = heavy::base::include_;
  HEAVY_BASE_VAR(include_ci)      = heavy::base::include_ci;
  HEAVY_BASE_VAR(include_library_declarations)
    = heavy::base::include_library_declarations;
  HEAVY_BASE_VAR(source_loc)      = heavy::base::source_loc;
  HEAVY_BASE_VAR(parse_source_file).init(Context);

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
  HEAVY_BASE_VAR(write)   = heavy::base::write;
  HEAVY_BASE_VAR(newline) = heavy::base::newline;
  HEAVY_BASE_VAR(eq)      = heavy::base::eqv;
  HEAVY_BASE_VAR(equal)   = heavy::base::equal;
  HEAVY_BASE_VAR(eqv)     = heavy::base::eqv;
  HEAVY_BASE_VAR(call_cc) = heavy::base::call_cc;
  HEAVY_BASE_VAR(with_exception_handler)
    = heavy::base::with_exception_handler;
  HEAVY_BASE_VAR(raise)   = heavy::base::raise;
  HEAVY_BASE_VAR(error)   = heavy::base::error;
  HEAVY_BASE_VAR(dynamic_wind) = heavy::base::dynamic_wind;

  HEAVY_BASE_VAR(eval)    = heavy::base::eval;
  HEAVY_BASE_VAR(op_eval) = heavy::base::op_eval;
  HEAVY_BASE_VAR(compile) = heavy::base::compile;
  HEAVY_BASE_VAR(module_path).init(Context);
}

// initializes the module and loads lookup information
// for the compiler
inline void HEAVY_BASE_LOAD_MODULE(heavy::Context& Context) {
  // Note: We call HEAVY_BASE_INIT in the constructor of Context.
  heavy::initModule(Context, HEAVY_BASE_LIB_STR, {
    // syntax
    {"define",        HEAVY_BASE_VAR(define)},
    {"define-syntax", HEAVY_BASE_VAR(define_syntax)},
    {"if",            HEAVY_BASE_VAR(if)},
    {"lambda",        HEAVY_BASE_VAR(lambda)},
    {"quasiquote",    HEAVY_BASE_VAR(quasiquote)},
    {"quote",         HEAVY_BASE_VAR(quote)},
    {"set!",          HEAVY_BASE_VAR(set)},
    {"syntax-rules",  HEAVY_BASE_VAR(syntax_rules)},
    {"begin",         HEAVY_BASE_VAR(begin)},
    {"cond-expand",   HEAVY_BASE_VAR(cond_expand)},
    {"define-library",HEAVY_BASE_VAR(define_library)},
    {"export",        HEAVY_BASE_VAR(export)},
    {"include",       HEAVY_BASE_VAR(include)},
    {"include-ci",    HEAVY_BASE_VAR(include_ci)},
    {"include-library-declarations",
      HEAVY_BASE_VAR(include_library_declarations)},
    {"source-loc",    HEAVY_BASE_VAR(source_loc)},
    {"parse-source-file",
                      HEAVY_BASE_VAR(parse_source_file).get(Context)},

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
    {"write",   HEAVY_BASE_VAR(write)},
    {"newline", HEAVY_BASE_VAR(newline)},
    {"eq?",     HEAVY_BASE_VAR(eq)},
    {"equal?",  HEAVY_BASE_VAR(equal)},
    {"eqv?",    HEAVY_BASE_VAR(eqv)},
    {"call/cc", HEAVY_BASE_VAR(call_cc)},
    {"with-exception-handler", HEAVY_BASE_VAR(with_exception_handler)},
    {"raise", HEAVY_BASE_VAR(raise)},
    {"error", HEAVY_BASE_VAR(error)},
    {"dynamic-wind", HEAVY_BASE_VAR(dynamic_wind)},

    {"eval",    HEAVY_BASE_VAR(eval)},
    {"op-eval", HEAVY_BASE_VAR(op_eval)},
    {"compile", HEAVY_BASE_VAR(compile)},
    {"module-path", HEAVY_BASE_VAR(module_path).get(Context)},
  });
}

}

namespace heavy::base {
/* InitParseSourceFile
 *  - Provide a hook for including source files by
 *    setting parse-source-file.
 *  - Inside Fn, the user can call HeavyScheme::ParseSourceFile
 *    to allow them to implement file lookup and storage
 *    and manage their own source locations.
 *  - Fn is a more documentative interface for the user
 *    that might not be interested in writing Syntax in C++.
 *  - Fn must still call C.Cont(..) or C.RaiseError(..).
 *
 *  using ParseSourceFileFn = void(heavy::Context&,
 *                                 heavy::SourceLocation,
 *                                 heavy::String*);
 */
template <typename ParseSourceFileFn>
void InitParseSourceFile(heavy::Context& C, ParseSourceFileFn Fn) {
  auto ParseFn = [Fn](heavy::Context& C, heavy::ValueRefs Args) {
    if (Args.size() != 2)
      return C.RaiseError("parse-source-file expects 2 arguments");
    heavy::SourceLocation Loc = Args[0].getSourceLocation();
    auto* Filename = dyn_cast<heavy::String>(Args[1]);
    if (!Filename)
      return C.RaiseError("expecting filename");
    Fn(C, Loc, Filename);
  };
  HEAVY_BASE_VAR(parse_source_file).set(C, C.CreateLambda(ParseFn));
}

}

namespace heavy::detail {
// Declare utility functions for iterating UTF-8 characters.
class Utf8View {
  llvm::StringRef Range;
public:
  Utf8View(llvm::StringRef StrView)
    : Range(StrView)
  { }

  bool empty() const {
    return Range.empty();
  }

  std::pair<uint32_t, unsigned> decode_front() const;

  // Return a single heavy::Char or nullptr on error.
  heavy::Value drop_front() {
    auto [codepoint, length] = decode_front();
    if (length == 0) return nullptr;

    Range = Range.substr(length);
    return heavy::Char(codepoint);
  }
};
void encode_utf8(uint32_t UnicodeScalarValue,
                 llvm::SmallVectorImpl<char> &Result);

// Convert to a hexadecimal string for
// writing character constants and escaped
// hexadecimal codes.
void encode_hex(uint32_t Code,
                llvm::SmallVectorImpl<char> &Result);

// from_hex
std::pair</*CodePoint=*/  uint32_t,
          /*IsError=*/    bool>
from_hex(llvm::StringRef Chars);

}

#endif

