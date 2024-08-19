//===--------- Parser.h - HeavyScheme Language Parser -----------*- C++ -*-===//
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

#ifndef LLVM_HEAVY_PARSER_H
#define LLVM_HEAVY_PARSER_H

#include "heavy/Context.h"
#include "heavy/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace heavy {

// TODO Rename to ParseResult
// TODO Determine if we need this since we have RaiseError.
class ValueResult {
  Value V = nullptr;

public:
  ValueResult() = default;
  ValueResult(Value V)
    : V(V)
  { }

  Value get() {
    assert(isUsable() && "should not get unusable result");
    return V;
  }

  bool isUsable() {
    return V && V.getKind() != ValueKind::Undefined;
  }
};

class Parser {
  using ValueResult = heavy::ValueResult;
  using Value = heavy::Value;
  heavy::Lexer& Lexer;
  heavy::Context& Context;
  Token Tok = {};
  TokenKind Terminator = tok::eof;
  bool IsFinished = false;
  SourceLocation PrevTokLocation;
  std::string LiteralResult = {};
  std::string ErrorMsg = {};
  Token ErrTok = {};

  auto ValueEmpty() { return ValueResult(); }
  auto ValueError() { return ValueResult(Context.CreateUndefined()); }
  bool ParseLiteralImpl();

  ValueResult ParseExpr();
  ValueResult ParseExprAbbrev(char const* Name);

  ValueResult ParseCharConstant();
  ValueResult ParseExternName();
  ValueResult ParseList(Token const& StartTok);
  ValueResult ParseListStart();
  ValueResult ParseNumber();
  ValueResult ParseString();
  ValueResult ParseSymbol();
  ValueResult ParseEscapedSymbol();
  ValueResult ParseVectorStart(bool IsByteVector = false);
  ValueResult ParseVector(llvm::SmallVectorImpl<Value>& Xs,
                          bool IsByteVector);

  ValueResult ParseDottedCdr(Token const& StartTok);
  ValueResult ParseSpecialEscapeSequence();

  ValueResult SetError(Token const& Tok, StringRef Msg) {
    ErrTok = Tok;
    ErrorMsg = Msg;
    // Prevent infinite loops in the absence of error checks.
    ConsumeToken();
    return ValueError();
  }

  bool CheckTerminator() {
    if (Tok.getKind() == Terminator) {
      IsFinished = true;
    }
    return IsFinished;
  }

public:
  Parser(heavy::Lexer& Lexer, heavy::Context& C)
    : Lexer(Lexer)
    , Context(C)
  { }

  bool isFinished() const {
    return IsFinished;
  }

  bool HasError() const {
    return !ErrorMsg.empty();
  }

  void RaiseError() {
    assert(HasError() && "There must be an error to raise.");
    Token TempTok = ErrTok.getLength() > 0 ? ErrTok : Tok;
    SourceLocation Loc = TempTok.getLocation();
    Context.setLoc(Loc);
    Context.RaiseError(ErrorMsg);
  }

  ValueResult ParseTopLevelExpr();
  ValueResult Parse(heavy::TokenKind Term = tok::eof);

  // Consumes the first token for parsing.
  // If the terminator is brace-like, the
  // matching opening brace is expected as
  // the first token and is consumed.
  // Returns false on failure. (sorry)
  bool PrimeToken(heavy::TokenKind Term) {
    Lexer.Lex(Tok);

    Terminator = Term;
    switch (Terminator) {
    case tok::r_brace:
      return TryConsumeToken(tok::l_brace, "expecting {");
    case tok::r_paren:
      return TryConsumeToken(tok::l_paren, "expecting (");
    case tok::r_square:
      return TryConsumeToken(tok::l_square, "expecting [");
    default:
      return true;
    }
  }

  SourceLocation ConsumeToken() {
    PrevTokLocation = Tok.getLocation();
    Lexer.Lex(Tok);
    return PrevTokLocation;
  }

  bool TryConsumeToken(heavy::TokenKind Expected, llvm::StringRef ErrMsg) {
    if (Tok.isNot(Expected)) {
      SetError(Tok, ErrMsg);
      return false;
    }
    ConsumeToken();
    return true;
  }
};

}  // end namespace heavy

#endif
