//===--- ParserHeavyScheme.h - HeavyScheme Language Parser ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Parser interface for HeavyScheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSER_HEAVY_SCHEME_H
#define LLVM_CLANG_PARSE_PARSER_HEAVY_SCHEME_H

#include "clang/AST/HeavyScheme.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include <string>

namespace clang {

class Parser;
class DeclContext;

// TODO Breakout the actual parsing stuff
//      that only depends on Token and SourceLocation
//      and maybe Preprocessor
class ParserHeavyScheme {
  using ValueResult = heavy::ValueResult;
  using Value = heavy::Value;
  Preprocessor& PP;
  heavy::Context& Context;
  Parser& CxxParser;
  Token Tok = {};
  SourceLocation PrevTokLocation;
  std::string LiteralResult = {};

  ValueResult ParseExpr();
  ValueResult ParseExprAbbrev(char const* Name);

  ValueResult ParseCharConstant();
  ValueResult ParseCppDecl();
  ValueResult ParseList(Token const& StartTok);
  ValueResult ParseListStart();
  ValueResult ParseNumber();
  ValueResult ParseString();
  ValueResult ParseSymbol();
  ValueResult ParseTypename();
  ValueResult ParseVectorStart();
  ValueResult ParseVector(SmallVectorImpl<Value*>& Xs);

  ValueResult ParseDottedCdr(Token const& StartTok);
  ValueResult ParseSpecialEscapeSequence();

  void SetError(Token& Tok, StringRef Msg) {
    SourceLocation Loc = Tok.getLocation();
    Context.SetError(Context.CreateError(Loc, Msg, Context.CreateEmpty()));
  }

public:
  ParserHeavyScheme(Preprocessor& PP, heavy::Context& C, Parser& P)
    : PP(PP)
    , Context(C)
    , CxxParser(P)
  {
    // Give the scheme context access to the
    // C++ parser
  }

  // Gets or creates an environment for a clang::DeclContext
  Value* LoadEmbeddedEnv(DeclContext*);

  ValueResult ParseTopLevelExpr();

  SourceLocation ConsumeToken() {
    PrevTokLocation = Tok.getLocation();
    PP.LexHeavyScheme(Tok);
    return PrevTokLocation;
  }

  bool TryConsumeToken(tok::TokenKind Expected) {
    if (Tok.isNot(Expected))
      return false;
    ConsumeToken();
    return true;
  }

#if 0 // TODO if we hoist to ParseDeclHeavy.cpp
  // Parses until the ending r_brace
  // and evaluates top level expressions.
  // Expects that the first token is heavy_scheme
  // Returns true if there was an error
  bool Parse();
#endif
};

}  // end namespace clang

#endif
