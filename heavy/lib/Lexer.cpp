//===------------ Lexer.cpp - HeavyScheme Language Lexer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Lexer
//
//===----------------------------------------------------------------------===//

#include "heavy/Lexer.h"
#include "heavy/Source.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

using namespace heavy;

namespace {
  bool isExtendedAlphabet(char c) {
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
    case '+': case '-': case '.': case '*': case '/': case '<': case '=':
    case '>': case '!': case '?': case ':': case '$': case '%': case '_':
    case '&': case '~': case '^':
      return true;
    }
    return false;
  }

  bool isDelimiter(char c) {
    switch(c) {
    case 0:
    case '"': case ';':
    case '(': case ')':
    case '[': case ']':
    case '{': case '}':
    case '|':
      return true;
    }

    if (clang::isWhitespace(c)) {
      return true;
    }

    return false;
  }
}

void Lexer::Lex(Token& Tok) {
  const char* CurPtr = BufferPtr;
  TokenKind Kind;
  do {
    ProcessWhitespace(CurPtr);
  } while (TryProcessComment(CurPtr));

  // Act on the current character
  char c = *CurPtr++;
  // These are all considered "initial characters".
  switch(c) {
  // Identifiers.
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
  // Identifiers (extended alphabet)
  case '*': case '/': case '<': case '=': case '>': case '!':
  case '?': case ':': case '$': case '%': case '_':
  case '&': case '~': case '^':
    return LexIdentifier(Tok, CurPtr);
  // Integer constants
  case '-': case '+':
    return LexNumberOrIdentifier(Tok, CurPtr);
  case '.':
    return LexNumberOrEllipsis(Tok, CurPtr);
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return LexNumber(Tok, CurPtr);
  case '#':
    return LexSharpLiteral(Tok, CurPtr);
  case '"':
    return LexStringLiteral(Tok, CurPtr);
  case '(':
    Kind = tok::l_paren;
    break;
  case ')':
    Kind = tok::r_paren;
    break;
  case '{':
    Kind = tok::l_brace;
    break;
  case '}':
    Kind = tok::r_brace;
    break;
  case '[':
    Kind = tok::l_square;
    break;
  case ']':
    Kind = tok::r_square;
    break;
  case '\'':
    Kind = tok::quote;
    break;
  case '`':
    Kind = tok::quasiquote;
    break;
  case ',': {
    if (*(CurPtr) == '@') {
      Kind = tok::unquote_splicing;
      ++CurPtr;
    } else {
      Kind = tok::unquote;
    }
    break;
  }
  case 0: {
    // don't go past the EOF
    --CurPtr;
    if (IsBlockComment) {
      Kind = tok::block_comment_eof;
    } else {
      Kind = tok::eof;
    }
    break;
  }
  default:
    Kind = tok::unknown;
    break;
  }

  // Handle the single character tokens
  FormTokenWithChars(Tok, CurPtr, Kind);
}

void Lexer::LexIdentifier(Token& Tok, const char *CurPtr) {
  bool IsInvalid = false;
  char c = *CurPtr;
  while (!isDelimiter(c)) {
    IsInvalid |= !isExtendedAlphabet(c);
    c = ConsumeChar(CurPtr);
  }

  if (IsInvalid) {
    return LexUnknown(Tok, CurPtr);
  }

  return FormIdentifier(Tok, CurPtr);
}

void Lexer::LexNumberOrIdentifier(Token& Tok, const char *CurPtr) {
  // + and - are valid characters by themselves
  assert(*(CurPtr - 1) == '+' ||
         *(CurPtr - 1) == '-');
  if (isDelimiter(*CurPtr)) {
    // '+' | '-' are valid identifiers
    return FormIdentifier(Tok, CurPtr);
  }
  // Lex as a number
  LexNumber(Tok, CurPtr);
}

void Lexer::LexNumberOrEllipsis(Token& Tok, const char *CurPtr) {
  const char *OrigPtr = CurPtr;
  // We already consumed a dot .
  char c1 = *CurPtr;
  if (isDelimiter(c1)) {
    return FormTokenWithChars(Tok, CurPtr, tok::period);
  }
  char c2 = ConsumeChar(CurPtr);
  char c3 = ConsumeChar(CurPtr);
  if (c1 == '.' && c2 ==  '.' && isDelimiter(c3)) {
    // '...' is a valid identifier (via <dot subsequent>)
    // TODO Implement <peculiar identifier>.
    return FormIdentifier(Tok, CurPtr);
  }
  // Lex as a number
  LexNumber(Tok, OrigPtr);
}

void Lexer::LexNumber(Token& Tok, const char *CurPtr) {
  // let the parser figure it out
  SkipUntilDelimiter(CurPtr);
  FormLiteral(Tok, CurPtr, tok::numeric_constant);
}

// These could be numbers, character constants, or other literals
// such as #t #f for true and false
void Lexer::LexSharpLiteral(Token& Tok, const char *CurPtr) {
  // We already consumed the #
  char c = *CurPtr++;
  // If we expect the token to end after
  // `c` then we set RequiresDelimiter
  bool RequiresDelimiter = false;
  TokenKind Kind;
  switch (c) {
  case '\\':
    SkipUntilDelimiter(CurPtr);
    return FormLiteral(Tok, CurPtr, tok::char_constant);
  case 't':
    Kind = tok::true_;
    RequiresDelimiter = true;
    break;
  case 'f':
    Kind = tok::false_;
    RequiresDelimiter = true;
    break;
  case '(':
    Kind = tok::vector_lparen;
    break;
  case ';':
    // The datum that follows this token
    // is treated as a comment but we still
    // need to parse the datum to know where it ends
    Kind = tok::comment_datum;
    break;
  case '_':
    ++CurPtr; // Consume the underscore.
    Kind = tok::extern_name;
    RequiresDelimiter = true;
    break;
  // unsupported radix R specifiers
  case 'd':
  case 'b': case 'o': case 'x':
  // unsupported exactness specifiers
  case 'i': case 'e':
  default:
    Kind = tok::unknown;
    SkipUntilDelimiter(CurPtr);
  }

  // We should be at a delimiter at this point or
  // we are dealing with something invalid
  if (RequiresDelimiter && !isDelimiter(*CurPtr)) {
    SkipUntilDelimiter(CurPtr);
    FormTokenWithChars(Tok, CurPtr, tok::unknown);
  } else {
    FormTokenWithChars(Tok, CurPtr, Kind);
  }
}

void Lexer::LexStringLiteral(Token& Tok, const char *CurPtr) {
  // Already consumed the "
  char c = *CurPtr;
  while (c != '"') {
    if (c == '\\') {
      // TODO Support R7RS escape sequences
      // currently we only escape arbitrary characters
      ConsumeChar(CurPtr);
    } else if (c == '\0') {
      FormTokenWithChars(Tok, --CurPtr, tok::string_literal_eof);
      return;
    }
    c = ConsumeChar(CurPtr);
  }
  // Consume the "
  ConsumeChar(CurPtr);
  FormLiteral(Tok, CurPtr, tok::string_literal);
}

void Lexer::LexUnknown(Token& Tok, const char *CurPtr) {
  SkipUntilDelimiter(CurPtr);
  FormTokenWithChars(Tok, CurPtr, tok::unknown);
}

void Lexer::SkipUntilDelimiter(const char *&CurPtr) {
  char c = *CurPtr;
  while (!isDelimiter(c)) {
    c = ConsumeChar(CurPtr);
  }
}

void Lexer::ProcessWhitespace(const char *&CurPtr) {
  // adds whitespace flags to Tok if needed
  char c = *CurPtr;

  if (!clang::isWhitespace(c)) {
    return;
  }

  while (true) {
    while (clang::isHorizontalWhitespace(c)) {
      c = ConsumeChar(CurPtr);
    }

    if (!clang::isVerticalWhitespace(c)) {
      break;
    }

    c = ConsumeChar(CurPtr);
  }

  BufferPtr = CurPtr;
}

void Lexer::ProcessBlockComment(const char *&CurPtr) {
  assert((*CurPtr == '#' && *(CurPtr + 1) == '|') &&
      "Expected block comment opening");
  auto BlockRaii = BlockCommentRaii(*this);
  CurPtr += 2;
  while (true) {
    if (*CurPtr == '#' && *(CurPtr + 1) == '|') {
      // these things can be nested
      ProcessBlockComment(CurPtr);
    }
    if (*CurPtr == '|' && *(CurPtr + 1) == '#') {
      // we found the matching end
      CurPtr += 2;
      break;
    } else if (*CurPtr == '\0') {
      BlockRaii.setInvalidEof();
      return;
    }
    ++CurPtr;
  }
}

bool Lexer::TryProcessComment(const char *&CurPtr) {
  char c = *CurPtr;
  if (c == ';') {
    // Skip until new line
    while (!clang::isVerticalWhitespace(c)) {
      c = ConsumeChar(CurPtr);
    }
    return true;
  } else if (c == '#' && *(CurPtr + 1) == '|') {
    ProcessBlockComment(CurPtr);
    return true;
  }

  return false;
}

// Copy/Pasted from Lexer (mostly)
SourceLocation Lexer::getSourceLocation(const char *Loc) const {
  assert(Loc >= BufferStart && Loc <= BufferEnd &&
         "Location out of range for this buffer!");

  // In the normal case, we're just lexing from a simple file buffer, return
  // the file id from FileLoc with the offset specified.
  unsigned CharNo = Loc - BufferStart;
  return FileLoc.getLocWithOffset(CharNo);
}
