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
#include "llvm/ADT/SmallVector.h"
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
  std::string TempString;
  if (Symbol* S = dyn_cast<Symbol>(Name)) {
    Str = S->getVal();
  } else if (String* S = dyn_cast<String>(Name)) {
    Str = S->getView();
  } else if (isa<Int>(Name) && cast<Int>(Name) >= 0) {
    int32_t I = cast<Int>(Name);
    TempString = std::to_string(I);
    Str = llvm::StringRef(TempString);
  } else {
    return setError("expected identifier or unsigned integer for name", Name);
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

char parseSpecialChar(llvm::StringRef Buf) {
  if (Buf.size() < 2)
    return 0;

  // It would be cool to generate switch using code from
  //  Buf[0] * 256 + Buf[1];

  Buf = Buf.take_front(2);
  if (Buf == llvm::StringRef("nt")) return '!';
  if (Buf == llvm::StringRef("dl")) return '$';
  if (Buf == llvm::StringRef("rm")) return '%';
  if (Buf == llvm::StringRef("ad")) return '&';
  if (Buf == llvm::StringRef("ml")) return '*';
  if (Buf == llvm::StringRef("pl")) return '+';
  if (Buf == llvm::StringRef("mi")) return '-';
  if (Buf == llvm::StringRef("dt")) return '.';
  if (Buf == llvm::StringRef("dv")) return '/';
  if (Buf == llvm::StringRef("cl")) return ':';
  if (Buf == llvm::StringRef("lt")) return '<';
  if (Buf == llvm::StringRef("eq")) return '=';
  if (Buf == llvm::StringRef("gt")) return '>';
  if (Buf == llvm::StringRef("qu")) return '?';
  if (Buf == llvm::StringRef("at")) return '@';
  if (Buf == llvm::StringRef("eo")) return '^';
  if (Buf == llvm::StringRef("co")) return '~';

  // User should parse CharHexCode.
  return 0;
}

std::string Mangler::mangleCharHexCode(Continuation Cont, Twine Prefix,
                                       llvm::StringRef Str) {
  llvm_unreachable("TODO mangle special characters");
  // we may drop multiple chars if we are supporting utf8
  return mangleNameSegment(Cont, "???", Str.drop_front(1));
}

bool Mangler::isExternalVariable(llvm::StringRef ModulePrefix,
                                 llvm::StringRef VarName) {
  if (!VarName.starts_with(ModulePrefix))
    return false;
  llvm::StringRef Result = parseModulePrefix(VarName);
  // VarName is external if it does NOT have the same prefix.
  return Result != ModulePrefix;
}

namespace {
bool consumeNameSegment(llvm::StringRef& Buf) {
  if (Buf.size() < 2) return false;
  if (Buf[0] == '0') {
    // TODO handle <hex-char-encoding>
    return false;
  } else if (Buf[0] >= 'a' && Buf[0] <= 'z') {
    // <special-char-code>
    Buf = Buf.drop_front(2);
  } else if (Buf[0] >= '1' && Buf[0] <= '9') {
    // <length encoding> S <id-segment>
    size_t Length = 0;
    while (Buf.size() > 0 && llvm::isDigit(Buf[0])) {
      unsigned Digit = llvm::hexDigitValue(Buf[0]);
      Length = Length * 10 + Digit;
      Buf = Buf.drop_front(1);
    }
    // Expect S.
    if (Buf[0] != 'S')
      return false;
    Buf = Buf.drop_front(1); // Consume 'S'.
    Buf = Buf.drop_front(Length);
  }
  return true;
}
}


// parseModulePrefix - Parse the module prefix of a valid symbol name.
//                     The prefix for an invalid name should be safe,
//                     but is undefined and may trigger assertions.
llvm::StringRef Mangler::parseModulePrefix(llvm::StringRef Name) {
  llvm::StringRef Ending = Name.drop_front(getManglePrefix().size());
  size_t PrevEndingSize = 0;
  while (Ending.size() > 0) {
    // Check for progress.
    if (Ending.size() == PrevEndingSize)
      return {};
    PrevEndingSize = Ending.size();

    switch (Ending[0]) {
      case 'V': case 'F': case 'A':
        // Break from the loop
        break;
      case 'L':
        // Library name
        Ending = Ending.drop_front(1);
        continue;
      case '_': {
        llvm::StringRef ModuleFile("__module_file");
        if (Ending.starts_with(ModuleFile))
          Ending.consume_front(ModuleFile);
        continue;
      }
      default:
        if (!consumeNameSegment(Ending))
          return {};  // Something was invalid.
        continue;
    }
    break;
  }
  return Name.drop_back(Ending.size());
}

bool Mangler::parseNameSegment(llvm::StringRef& Input,
                               llvm::SmallVectorImpl<char>& Output) {
  if (Input.size() < 2) return false;
  if (Input[0] == '0') {
    llvm_unreachable("TODO handle hex-char-encoding");
  } else if (Input[0] >= 'a' && Input[0] <= 'z') {
    // <special-char>
    Output.push_back(parseSpecialChar(Input));
    Input = Input.drop_front(2);
  } else if (Input[0] >= '1' && Input[0] <= '9') {
    // <length encoding> S <id-segment>
    size_t Length = 0;
    while (Input.size() > 0 && llvm::isDigit(Input[0])) {
      unsigned Digit = llvm::hexDigitValue(Input[0]);
      Length = Length * 10 + Digit;
      Input = Input.drop_front(1);
    }
    // Expect S.
    if (Input[0] != 'S')
      return {};
    Input = Input.drop_front(1); // Consume 'S'.
    llvm::StringRef ToOutput = Input.take_front(Length);
    Output.append(ToOutput.begin(), ToOutput.end());
    Input = Input.drop_front(Length);
  }

  return true;
}

bool Mangler::parseLibraryName(llvm::StringRef& Input, 
                               llvm::SmallVectorImpl<char>& Output) {
  if (Input.empty() || Input[0] != 'L')
    return false;

  // Drop the L
  Input = Input.drop_front(1);
  while (!Input.empty() && Input[0] != 'L')
    if (!parseNameSegment(Input, Output))
      return false;
  return true;
}

}

