//===- Base.h - Base library functions for HeavyScheme ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares values and functions for the (heavy builtins) scheme library
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_BUILTINS_H
#define LLVM_HEAVY_BUILTINS_H

#include "heavy/Context.h"
#include "heavy/Value.h"
#include "llvm/ADT/SmallVector.h"

#define HEAVY_BASE_LIB                _HEAVYL5SheavyL8Sbuiltins
#define HEAVY_BASE_LIB_(NAME)         _HEAVYL5SheavyL8Sbuiltins ## NAME
#define HEAVY_BASE_LIB_STR            "_HEAVYL5SheavyL8Sbuiltins"
#define HEAVY_BASE_LOAD_MODULE        HEAVY_BASE_LIB_(_load_module)
#define HEAVY_BASE_INIT               HEAVY_BASE_LIB_(_init)

#define HEAVY_IMPORT_VAR              "_HEAVY_import"
#define HEAVY_LOAD_MODULE_VAR         "_HEAVY_load_module"

#define HEAVY_BASE_VAR(NAME)          ::heavy::builtins_var::NAME
#define HEAVY_BASE_VAR_STR(NAME)      HEAVY_BASE_VAR_STR__##NAME
#define HEAVY_BASE_VAR_STR__error     HEAVY_BASE_LIB_STR "V5Serror"

// Forward declare vars that are used in the C++ codebase.
namespace heavy::builtins_var {
extern heavy::ContextLocal module_path;
extern heavy::ContextLocal parse_source_file;
extern heavy::ExternFunction compile;
extern heavy::ExternFunction eval;
extern heavy::ExternFunction op_eval;
}

namespace heavy::builtins {
void eval(Context& C, ValueRefs Args);
void import_(Context& C, ValueRefs Args);
void load_module(Context& C, ValueRefs Args);
}

extern "C" {
// Initialize the module for run-time independent of the compiler.
void HEAVY_BASE_INIT(heavy::Context& Context);

// Initialize the module and loads lookup information
// for the compiler.
void HEAVY_BASE_LOAD_MODULE(heavy::Context& Context);
}

namespace heavy::builtins {
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
