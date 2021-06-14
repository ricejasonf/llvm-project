//===------- Mangle.cpp - HeavyScheme Mangler Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for name mangling of scheme libraries, variables,
// and functions
//
//===----------------------------------------------------------------------===//

#include "heavy/Value.h"
#include "heavy/Mangle.h"

namespace {
bool isSpecialChar(char c) {
  // TODO
  // We could make a table similar to clang::charinfo::InfoTable
  // for more efficient processing here and possibly elsewhere.
  switch(c) {
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
  case '_':
    return false;
  }
  return true;
}
}

namespace heavy {

std::string Mangler::mangleModule(Value Spec) {
  return mangleModuleName(ManglePrefix, Spec);
}

std::string Mangler::mangleModuleName(Twine Prefix, Value Spec) {
  if (isa<Empty>(Spec)) return Prefix.str();
  Pair* P = dyn_cast<Pair>(Spec);
  if (!P) return setError("expected list", Spec);

  auto Cont = [&](Twine Result) {
    return mangleModuleName(Result, P->Cdr);
  };
  return mangleName(Cont, Prefix + Twine('L'), P->Car);
}

std::string Mangler::mangleVariable(Twine ModulePrefix, Value Name) {
  auto Cont = [&](Twine Result) { return Result.str(); };
  return mangleName(Cont, ModulePrefix + Twine('V'), Name);
}

std::string Mangler::mangleSpecialName(Twine ModulePrefix,
                                       llvm::StringRef Name) {
  assert(Name.drop_until(isSpecialChar).empty() &&
      "mangled special name must be valid C identifier");
  Twine Prefix = ModulePrefix + Twine('_');
  return Prefix.concat(Name).str();
}

std::string Mangler::mangleName(Continuation Cont, Twine Prefix,
                                Value Name) {
  llvm::StringRef Str = llvm::StringRef();
  if (Symbol* S = dyn_cast<Symbol>(Name)) {
    Str = S->getVal();
  } else if (String* S = dyn_cast<String>(Name)) {
    Str = S->getView();
  } else {
    return setError("expected name in name mangler", Name);
  }

  // <empty-name>
  if (Str.empty()) {
    return Prefix.concat(Twine('N')).str();
  }

  return mangleNameSegment(Cont, Prefix, Str);
}

std::string Mangler::mangleNameSegment(Continuation Cont, Twine Prefix,
                                       llvm::StringRef Str) {
  if (Str.empty()) return Cont(Prefix);
  llvm::StringRef NameSegment = Str.take_until(isSpecialChar);
  if (NameSegment.empty()) return mangleSpecialChar(Cont, Prefix, Str);
  size_t NameSegmentSize = NameSegment.size(); // must be on stack
  Twine NameSegmentPrefix = Twine(NameSegmentSize) + Twine('S');
  Twine FullPrefix = Prefix + NameSegmentPrefix;
  return mangleNameSegment(Cont, 
                           FullPrefix + NameSegment,
                           Str.drop_front(NameSegment.size()));
}

std::string Mangler::mangleSpecialChar(Continuation Cont, Twine Prefix,
                                       llvm::StringRef Str) {
  if (Str.empty()) return Cont(Prefix); 
  llvm::StringRef Result;
  switch(Str.front()) {
  case '!': Result = "nt"; break;
  case '$': Result = "dl"; break;
  case '%': Result = "rm"; break;
  case '&': Result = "ad"; break;
  case '*': Result = "ml"; break;
  case '+': Result = "pl"; break;
  case '-': Result = "mi"; break;
  case '.': Result = "dt"; break;
  case '/': Result = "dv"; break;
  case ':': Result = "cl"; break;
  case '<': Result = "lt"; break;
  case '=': Result = "eq"; break;
  case '>': Result = "gt"; break;
  case '?': Result = "qu"; break;
  case '@': Result = "at"; break;
  case '^': Result = "eo"; break;
  case '~': Result = "co"; break;

  default:
    return mangleCharHexCode(Cont, Prefix, Str);
  }

  return mangleNameSegment(Cont, Prefix + Result, Str.drop_front(1));
}

std::string Mangler::mangleCharHexCode(Continuation Cont, Twine Prefix,
                                       llvm::StringRef Str) {
  llvm_unreachable("TODO mangle special characters");
  // we may drop multiple chars if we are supporting utf8
  return mangleNameSegment(Cont, "???", Str.drop_front(1));
}


}

