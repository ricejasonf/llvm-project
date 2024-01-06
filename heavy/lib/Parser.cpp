//===--- Parser.cpp - HeavyScheme Language Parser --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Parser for HeavyScheme.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/Context.h"
#include "heavy/Lexer.h"
#include "heavy/Parser.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using heavy::Context;
using heavy::Parser;
using heavy::ValueResult;
using llvm::StringRef;

namespace {
  // Returns true on an invalid number prefix notation
  bool parseNumberPrefix(const char*& CurPtr,
                         std::optional<bool>& IsExact,
                         std::optional<unsigned>& Radix) {
    if (*CurPtr != '#')
      return false;

    ++CurPtr;
    char c = *CurPtr;
    if (Radix.has_value() ||
        (IsExact.has_value() &&
         (c == 'e' || c == 'i'))) {
      return true;
    }

    switch (c) {
    case 'e':
      IsExact = true;
      break;
    case 'i':
      IsExact = false;
      break;
    case 'b':
      Radix = 2;
      break;
    case 'o':
      Radix = 8;
      break;
    case 'd':
      Radix = 10;
      break;
    case 'h':
      Radix = 16;
      break;
    default:
      return true;
    }
    ++CurPtr;
    return parseNumberPrefix(CurPtr, IsExact, Radix);
  }

  std::optional<llvm::APInt>
  tryParseInteger(StringRef TokenSpan, unsigned BitWidth, unsigned Radix) {
    // largely inspired by NumericLiteralParser::GetIntegerValue
    llvm::APInt RadixVal(BitWidth, Radix, /*IsSigned=*/true);
    llvm::APInt DigitVal(BitWidth, 0, /*IsSigned=*/true);
    llvm::APInt Val(BitWidth, 0, /*IsSigned=*/true);
    llvm::APInt OldVal = Val;
    bool Negate = false;
    bool OverflowOccurred = false;

    // may start with sign
    if (TokenSpan[0] == '+' || TokenSpan[0] == '-') {
      Negate = TokenSpan[0] == '-';
      TokenSpan = TokenSpan.drop_front(1);
    }
    for (char c : TokenSpan) {
      unsigned Digit = llvm::hexDigitValue(c);
      if (Digit >= Radix)
        return {};
      DigitVal = Digit;
      OldVal = Val;

      Val *= RadixVal;
      // The inverse operation should be the same or
      // it overflowed
      OverflowOccurred |= Val.udiv(RadixVal) != OldVal;
      Val += DigitVal;
      OverflowOccurred |= Val.ult(DigitVal);
    }
    if (OverflowOccurred)
      return {};
    if (Negate)
      Val.negate();
    return Val;
  }
}

ValueResult Parser::ParseTopLevelExpr() {
  if (CheckTerminator()) {
    return ValueEmpty();
  }
  return ParseExpr();
}

ValueResult Parser::Parse(heavy::TokenKind Term) {
  if (CheckTerminator()) {
    return ValueEmpty();
  }
  PrimeToken(Term);
  // ParseList uses the terminator token when
  // supplied with tok::eof.
  Token TempTok{{}, tok::eof};
  ValueResult Result = ParseList(TempTok);
  IsFinished = true;
  return Result;
}

ValueResult Parser::ParseExpr() {
  switch (Tok.getKind()) {
  case tok::l_paren:
    return ParseListStart();
  case tok::vector_lparen:
    return ParseVectorStart();
  case tok::numeric_constant:
    return ParseNumber();
  case tok::identifier:
    return ParseSymbol();
  case tok::extern_name:
    return ParseExternName();
  case tok::char_constant:
    return ParseCharConstant();
  case tok::true_: {
    ConsumeToken();
    return Value(Bool(true));
  }
  case tok::false_: {
    ConsumeToken();
    return Value(Bool(false));
  }
  case tok::string_literal:
    return ParseString();
  case tok::quote:
    return ParseExprAbbrev("quote");
  case tok::quasiquote:
    return ParseExprAbbrev("quasiquote");
  case tok::unquote:
    return ParseExprAbbrev("unquote");
  case tok::unquote_splicing:
    return ParseExprAbbrev("unquote-splicing");
  case tok::r_paren: {
    SetError(Tok, "extraneous closing paren (')')");
    CheckTerminator();
    ConsumeToken();
    return ValueError();
  }
  case tok::r_square: {
    SetError(Tok, "extraneous closing bracket (']')");
    CheckTerminator();
    ConsumeToken();
    return ValueError();
  }
  case tok::r_brace: {
    // extraneous brace should end parsing
    SetError(Tok, "extraneous closing brace ('}')");
    CheckTerminator();
    ConsumeToken();
    return ValueEmpty();
  }
  case tok::comment_datum: {
    // the expr that immediately follows the
    // #; token is discarded (commented)
    ConsumeToken();
    ParseExpr();
    return ParseExpr();
  }
  case tok::eof: {
    // TODO Track the start token of the current
    //      list being parsed if any and note it
    //      in the diagnostic output
    SetError(Tok, "unexpected end of file");
    IsFinished = true;
    return ValueError();
  }
  case tok::string_literal_eof: {
    SetError(Tok, "unterminated string literal");
    IsFinished = true;
    return ValueError();
  }
  case tok::block_comment_eof: {
    SetError(Tok, "unterminated block comment");
    IsFinished = true;
    return ValueError();
  }
  default: {
    SetError(Tok, "expected expression");
    ConsumeToken();
    return ValueError();
  }
  }
}

// ParseExprAbbrev - Normalizes abbreviated prefix notation to
//                   their equivalent syntax e.g. (quote expr)
ValueResult Parser::ParseExprAbbrev(char const* Name) {
  Token Abbrev = Tok;
  ConsumeToken();
  ValueResult Result = ParseExpr();
  if (!Result.isUsable()) return Result;

  Value S = Context.CreateSymbol(Name, Abbrev.getLocation());
  Value P = Context.CreatePair(S, Context.CreatePair(Result.get()));
  return P;
}

ValueResult Parser::ParseListStart() {
  // Consume the l_paren
  assert(Tok.is(tok::l_paren));
  Token StartTok = Tok;
  ConsumeToken();
  return ParseList(StartTok);
}

ValueResult Parser::ParseList(Token const& StartTok) {
  Token CurTok = Tok;
  // TODO Use StartTok to identify the proper
  //      closing token to match with

  // discard commented exprs
  while (Tok.is(tok::comment_datum)) {
    ConsumeToken();
    ParseExpr();
  }

  if ((StartTok.is(tok::l_paren) && Tok.is(tok::r_paren)) ||
      (StartTok.is(tok::eof) && Tok.is(Terminator))) {
    ConsumeToken();
    return Value(Empty{});
  }

  ValueResult Car = ParseExpr();
  if (!Car.isUsable()) return Car;

  ValueResult Cdr;
  if (Tok.is(tok::period)) {
    Cdr = ParseDottedCdr(StartTok);
  } else {
    Cdr = ParseList(StartTok);
  }

  if (!Cdr.isUsable()) return Cdr;

  return Value(
      Context.CreatePairWithSource(Car.get(),
                                   Cdr.get(),
                                   CurTok.getLocation()));
}

// We have a dot while parsing a list,
// so we expect a single expression
// then the closing r_paren
ValueResult Parser::ParseDottedCdr(Token const& StartTok) {
  assert(Tok.is(tok::period));
  ConsumeToken();
  ValueResult Cdr = ParseExpr();
  if (!TryConsumeToken(tok::r_paren, "invalid dot notation")) {
    return ValueError();
  }
  return Cdr;
}

ValueResult Parser::ParseVectorStart() {
  // consume the heavy_vector_lparen
  ConsumeToken();
  llvm::SmallVector<Value, 16> Xs;
  return ParseVector(Xs);
}

ValueResult Parser::ParseVector(llvm::SmallVectorImpl<Value>& Xs) {
  // discard commented exprs
  while (Tok.is(tok::comment_datum)) {
    ConsumeToken();
    ParseExpr();
  }
  if (Tok.is(tok::r_paren)) {
    ConsumeToken();
    return Value(Context.CreateVector(Xs));
  }
  ValueResult Result = ParseExpr();
  if (!Result.isUsable()) return Result;

  Xs.push_back(Result.get());
  return ParseVector(Xs);
}

ValueResult Parser::ParseCharConstant() {
  llvm_unreachable("TODO");
}

ValueResult Parser::ParseNumber() {
  char const* Current = Tok.getLiteralData().begin();
  char const* End     = Tok.getLiteralData().end();
  int BitWidth = 32; // Int uses int32_t
  std::optional<bool> IsExactOpt;
  std::optional<unsigned> RadixOpt;
  std::optional<llvm::APInt> IntOpt;

  if (parseNumberPrefix(Current, IsExactOpt, RadixOpt)) {
    llvm_unreachable("TODO diagnose invalid number prefix");
    return ValueError();
  }

  StringRef TokenSpan(Current, End - Current);
  bool IsExact = IsExactOpt.value_or(true);
  unsigned Radix = RadixOpt.value_or(10);

  // TODO use llvm::StringRef's integer parsing stuffs
  //      and just use a fixed BitWidth
  IntOpt = tryParseInteger(TokenSpan, BitWidth, Radix);

  ConsumeToken();

  if (IsExact && IntOpt.has_value()) {
    return Value(Int(IntOpt.value().getZExtValue()));
  }

  llvm::APFloat FloatVal(0.0f);
  if (IntOpt.has_value()) {
    llvm::APInt Int = IntOpt.value();
    FloatVal.convertFromAPInt(Int, /*isSigned=*/true,
                           llvm::APFloat::rmNearestTiesToEven);
  } else {
    auto Result = FloatVal.convertFromString(
        TokenSpan, llvm::APFloat::rmNearestTiesToEven);
    if (!Result) {
      llvm_unreachable("TODO invalid numerical syntax");
      return ValueError();
    }
  }
  return Value(Context.CreateFloat(FloatVal));
}

ValueResult Parser::ParseString() {
  // the literal must include the ""
  assert(Tok.getLength() >= 2);
  llvm::StringRef TokenSpan = Tok.getLiteralData()
    .substr(1, Tok.getLength() - 2);
  LiteralResult.clear();
  while (TokenSpan.size() > 0) {
    char c = TokenSpan[0];
    if (c == '\\') {
      TokenSpan = TokenSpan.drop_front();
      assert(TokenSpan.size() > 0 &&
        "string literal should be properly formed");
      switch (TokenSpan[0]) {
        case '\\':
        case '"':
        case '|':
          c = TokenSpan[0];
          break;
        case 'a':
          c = '\a';
          break;
        case 'b':
          c = '\b';
          break;
        case 't':
          c = '\t';
          break;
        case 'n':
          c = '\n';
          break;
        case 'r':
          c = '\r';
          break;
        case 'x': {
          TokenSpan = TokenSpan.drop_front();
          unsigned SemiColonPos = TokenSpan.find(';');
          if (SemiColonPos == llvm::StringRef::npos) {
            SetError(Tok, "char hex scalar value missing terminating semicolon");
            ConsumeToken();
            return ValueError();
          } else {
            llvm::SmallVector<char, 4> ByteSequence;
            llvm::StringRef HexCode = TokenSpan.slice(0, SemiColonPos);
            auto [HexValue, IsError] = detail::from_hex(HexCode);
            if (!IsError) {
              detail::encode_utf8(HexValue, ByteSequence);
              llvm::append_range(LiteralResult, ByteSequence);
              TokenSpan = TokenSpan.drop_front(SemiColonPos + 1);
              continue;
            }
            SetError(Tok, "invalid utf8 codepoint");
            ConsumeToken();
            return ValueError();
          }
          break;
        }
      }
      TokenSpan = TokenSpan.drop_front();
    } else if (TokenSpan.consume_front("\r\n")) {
      // normalize source-file newlines
      c = '\n';
    } else {
      // consume one char
      TokenSpan = TokenSpan.substr(1);
    }
    LiteralResult.push_back(c);
  }
  ConsumeToken();
  return Value(Context.CreateString(StringRef(LiteralResult)));
}

ValueResult Parser::ParseSymbol() {
  StringRef Str = Tok.getLiteralData();
  SourceLocation Loc = Tok.getLocation();
  ConsumeToken();
  return Value(Context.CreateSymbol(Str, Loc));
}

ValueResult Parser::ParseExternName() {
  StringRef Str = Tok.getLiteralData();
  SourceLocation Loc = Tok.getLocation();
  ConsumeToken();
  return Value(Context.CreateExternName(Loc, Str));
}
