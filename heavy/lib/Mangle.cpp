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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

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
  if (!P) return setError("expected library name", Spec);

  auto Cont = [&](Twine Result) {
    return mangleModuleName(Result, P->Cdr);
  };
  return mangleName(Cont, Prefix + Twine('L'), P->Car);
}

std::string Mangler::mangleVariable(Twine ModulePrefix, Value Name) {
  auto Cont = [&](Twine Result) { return Result.str(); };
  return mangleName(Cont, ModulePrefix + Twine('V'), Name);
}

std::string Mangler::mangleVariable(Twine ModulePrefix, llvm::StringRef Name) {
  auto Cont = [&](Twine Result) { return Result.str(); };
  return mangleName(Cont, ModulePrefix + Twine('V'), Name);
}

std::string Mangler::mangleFunction(Twine ModulePrefix, llvm::StringRef Name) {
  auto Cont = [&](Twine Result) { return Result.str(); };
  return mangleName(Cont, ModulePrefix + Twine('F'), Name);
}

std::string Mangler::mangleAnonymousId(Twine ModulePrefix, size_t Id) {
  auto Cont = [&](Twine Result) { return Result.str(); };
  Twine Name = Twine('A') + Twine(Id);
  return Cont(ModulePrefix + Name);
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
    return setError("expected identifier for name", Name);
  }
  return mangleName(Cont, Prefix, Str);
}

std::string Mangler::mangleName(Continuation Cont, Twine Prefix,
                                llvm::StringRef Str) {
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

bool Mangler::isExternalVariable(llvm::StringRef ModulePrefix,
                                 llvm::StringRef VarName) {
  if (!VarName.startswith(ModulePrefix))
    return false;
  llvm::StringRef Result = parseModulePrefix(VarName);
  return !Result.equals(ModulePrefix);
}

// parseModulePrefix - Parse the module prefix of a valid symbol name.
//                     The prefix for an invalid name should be safe,
//                     but is undefined and may trigger assertions.
llvm::StringRef Mangler::parseModulePrefix(llvm::StringRef Name) {
  llvm::StringRef Ending = Name.drop_front(getManglePrefix().size());
  while (Ending.size() > 0) {
    switch (Ending[0]) {
      case 'V': case 'F': case 'A':
        // Break from the loop
        break;
      case 'L':
        // Library name
        Ending = Ending.drop_front(1);
        continue;
      case '1': case '2': case '3': case '4': case '5':
      case '6': case '7': case '8': case '9':
        consumeNameSegment(Ending);
        continue;
      // TODO Handle CharHexCode
      default:
        // SpecialChar code is always two characters.
        Ending = Ending.drop_front(2);
        continue;
    }
    break;
  }
  return Name.drop_back(Ending.size());
}

// consumeNameSegment - Consume a valid name segment from Buf and
//                      return the parsed name.
//                      Invalid formats should be safe, but have a 
//                      result that is undefined and may trigger
//                      assertions.
llvm::StringRef Mangler::consumeNameSegment(llvm::StringRef& Buf) {
  assert(Buf[0] >= '1' && Buf[0] <= '9' &&
      "name segment should begin with length");
  llvm::StringRef Result = "";
  size_t Length = 0;
  while (Buf.size() > 0 && llvm::isDigit(Buf[0])) {
    unsigned Digit = llvm::hexDigitValue(Buf[0]);
    Length = Length * 10 + Digit;
    Buf = Buf.drop_front(1);
  }
  assert(Buf[0] == 'S' && "invalid name segment");
  Buf = Buf.drop_front(1); // Consume 'S'.
  Result = Buf.take_front(Length);
  Buf = Buf.drop_front(Length);
  return Result;
}

}

