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

#include "heavy/Context.h"
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

 Scheme allows arbitrary characters in identifiers using the
 | delimiter include a zero-length identifier via ||.

 The ability to demangle may also be desirable.

  <mangled-name>        ::= _HEAVY <encoding>

  <encoding>            ::= <module-name>
                        ::= <variable-name>
                        ::= <function-name>
                        ::= <special-name>

  <module-name>         ::= <module-name> <module-name-node>
  <module-name-node>    ::= L <name>

  <function-name>       ::= <module-name> F <name>
                        ::= F <name>

  # anonymous functions for lambdas
  <anonymous-name>      ::= <module-name> A <id-segment>
                        ::= A <id-segment>

  <variable-name>       ::= <module-name> V <name>
                        ::= V <name>

  <special-name>        ::= <module-name> _ <id-segment>

  <name>                ::= <name-encoding>
                        ::= <empty-name>

  <name-encoding>       ::= <name-encoding> <name-segment>
                        ::= <name-segment>

  <name-segment>        ::= <length-encoding> S <id-segment>
                        ::= <special-char>

  <special-char>        ::= <special-char-code>
                        ::= <hex-char-encoding>

  <hex-char-encoding>   ::= 0x <hex-char-code> z

  <empty-name>          ::= N
  <id-segment>          ::= [_A-Za-z0-9]+   # note these can start with digits
  <length-encoding>     ::= [1-9] [0-9]+

  <special-char-code>   ::= [a-z][a-z]      # see table for valid codes
  <hex-char-code>       ::= [0-9a-f]+
********************************************/

namespace heavy {

class Mangler {
  using Twine = llvm::Twine;
  using StringRef = llvm::StringRef;
  using Continuation = llvm::function_ref<std::string(Twine)>;
  static constexpr char const* ManglePrefix = "_HEAVY";

  heavy::Context& Context;
  llvm::StringRef NameBuffer = {};

  template <typename ...Args>
  std::string setError(Args... args) {
    Context.SetError(args...);
    return std::string{};
  }

  // This continuation passing style with twines could easily have
  // been a preallocated string buffer.
  std::string mangleModuleName(Twine Prefix, Value Name);
  std::string mangleName(Continuation, Twine Prefix, Value Name);
  std::string mangleName(Continuation, Twine Prefix, llvm::StringRef);
  std::string mangleNameSegment(Continuation, Twine Prefix, StringRef);
  std::string mangleSpecialChar(Continuation, Twine Prefix, StringRef);
  std::string mangleCharHexCode(Continuation, Twine Prefix, StringRef);

public:
  Mangler(heavy::Context& C)
    : Context(C)
  { }

  static llvm::StringRef getManglePrefix() { return ManglePrefix; }
  std::string mangleModule(Value Spec);
  std::string mangleVariable(Twine ModulePrefix, Value Name);
  std::string mangleFunction(Twine ModulePrefix, llvm::StringRef Name);
  std::string mangleSpecialName(Twine ModulePrefix, llvm::StringRef Name);

  // mangleAnonymousId - Id is meant to be strictly monotonic but is 
  //                     managed external to this class
  std::string mangleAnonymousId(Twine ModulePrefix, size_t Id);
};

}

#endif
