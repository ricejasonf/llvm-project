//===------ Mangle.h - Classes for mangling scheme names ------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions for mangling names for HeavyScheme
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_MANGLE_H
#define LLVM_HEAVY_MANGLE_H

#include "heavy/Value.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include <string>

/*******************************************
 The mangling rules are provided with this format
 copying the BNF rules from the Itanium ABI name mangling
 except that terminals may have a series of bracket enclosed
 regexes which may have a trailing + or *.

 The purpose is to provide unique names for libraries
 and global variables across libraries as well as external
 functions to facilitate the run-time's library specific
 routines such as module initilization.

 The ability to demangle is also desirable.

  <mangled-name>      ::= _HEAVY <encoding>

  <encoding>          ::= <module-name>
                      ::= <variable-name>
                      ::= <special-function>

  <module-name>       ::= L <name-encoding>
  <variable-name>     ::= <module-name> V <name-encoding>
  <special-function>  ::= <module-name> _ <name-segment>

  <name-prefix>       ::= <name-encoding>

  <name-encoding>     ::= <name-prefix> <name-encoding>
                      ::= <length-encoding> <name-segment>
                      ::= <special-char-code>
                      ::= <hex-char-encoding>

  <hex-char-encoding> :: = 0x <hex-char-code> z  # terminated by 'z'

  <name-segment>      ::= [_A-Za-z0-9]+
  <length-encoding>   ::= [1-9] [0-9]+

  <special-char-code> ::= [a-z][a-z]             # see table for valid codes
  <hex-char-code>     ::= [0-9a-f]+
********************************************/

namespace heavy {

class Mangler {
  using Twine = llvm::Twine;
  heavy::Context& Context;
  static constexpr llvm::StringRef HeavyPrefix = "_HEAVY";

  template <typename ...Args>
  Twine setError(Args... args) {
    Context.SetError(args...);
    return Twine();
  }

  llvm::StringRef getSpecialCharCode(char X) {
    switch(X) {
    case '!': return "nt";
    case '$': return "dl";
    case '%': return "rm";
    case '&': return "ad";
    case '*': return "ml";
    case '+': return "pl";
    case '-': return "mi";
    case '.': return "dt";
    case '/': return "dv";
    case ':': return "cl";
    case '<': return "lt";
    case '=': return "eq";
    case '>': return "gt";
    case '?': return "qu";
    case '@': return "at";
    case '^': return "eo";
    case '~': return "co";
    default:
      return llvm::StringRef();
    }
  }

  Twine mangleName(Value Name) {
    llvm::StringRef Str = llvm::StringRef();
    if (Symbol* S = dyn_cast<Symbol>(Name)) {
      Str = S->getVal();
    } else if (String* S = dyn_cast<String>(Name)) {
      Str = S->getView();
    }

    if (Str.empty()) return setError("expected name in name mangler", Name);
    // TODO split by special characters
    // encode each name segment with a decimal number followed by the segment
    // "extended identifier characters" will be a two letter code appearing
    // between name segments
    // 
    // every other character will use a specific two letter code followed by
    // a fixed length unicode hex code
  }

public:
  Mangler(heavy::Context& C)
    : Context(C)
  { }

  Twine mangleModule(Value Spec) {
    auto Result = Twine(Twine(HeavyPrefix, Twine('L')));
    Value Current = Spec;
    while (Pair* P = dyn_cast<Pair>(Current)) {
      Result += mangleName(P->Car);
      Current = P->Cdr;
    }
    return Result; 
  }

  Twine mangleVariable(Twine ModulePrefix, Value Name) {
    auto Prefix = Twine(ModulePrefix, Twine('V'));
    return Twine(Prefix, mangleName(Name));
  }

  // mangleSpecialName - Special names are used for functions or
  //                     variables that are not a part of scheme
  //                     but are used by the run-time to do stuff
  //                     like initialize a module. The input name
  //                     should be valid as a C identifier and
  //                     delimited by an underscore.
  Twine mangleSpecialName(Twine ModulePrefix, llvm::StringRef Name) {
    auto Prefix = Twine(ModulePrefix, Twine('V'));
    return Twine(Prefix, Twine(Name));
  }
}

}

#endif
