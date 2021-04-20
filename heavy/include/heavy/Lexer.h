//===------------ Lexer.h - Heavy Scheme Lexer ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Lexer interface for HeavyScheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_LEXER_H
#define LLVM_HEAVY_LEXER_H

#include "heavy/Source.h"
#include <cassert>
#include <cstdint>

namespace clang {
  class SourceLocation;
}

namespace heavy {

enum class TokenKind {
  unknown = 0,
  block_comment_eof,        // #| ... EOF (only used for unexpected EOF)
  char_constant,
  comment_datum,            // #;
  eof,
  false_,
  identifier,
  l_brace,
  l_paren,
  l_square, 
  numeric_constant,
  period,
  quasiquote,               // `
  quote,                    // '
  r_brace,
  r_paren,
  r_square,
  string_literal,
  string_literal_eof,
  true_,
  unquote,                  // ,
  unquote_splicing,         // ,(
  vector_lparen,            // #(
};

using tok = TokenKind;

struct Token {
  SourceLocation Loc;
  TokenKind Kind;
  llvm::StringRef LiteralData;

  SourceLocation getLocation() const { return Loc; }
  TokenKind getKind() const { return Kind; }

  bool is(TokenKind K)    const { return Kind == K; }
  bool isNot(TokenKind K) const { return Kind != K; }

  unsigned getLength() const { return LiteralData.size(); }
  llvm::StringRef getLiteralData() const { return LiteralData; }
};

class EmbeddedLexer {
protected:
  SourceLocation FileLoc;
  char const* BufferStart = nullptr;
  char const* BufferEnd   = nullptr;
  char const* BufferPtr   = nullptr;

  EmbeddedLexer(SourceLocation FileLoc, llvm::StringRef FileBuffer,
                char const* BufferPos)
    : FileLoc(FileLoc),
      BufferStart(FileBuffer.begin()),
      BufferEnd(FileBuffer.end()),
      BufferPtr(BufferPos)
  { 
    assert((BufferPtr >= BufferStart && BufferPtr < BufferEnd) &&
        "BufferPtr must be in range");
  }
public:
  unsigned GetByteOffset() {
    if (BufferPtr > BufferEnd)
      return BufferEnd - BufferStart;
    else
      return BufferPtr - BufferStart;
  }
};

class Lexer : public EmbeddedLexer {
  // IsBlockComment - block comments can be
  // nested so if we hit an eof the Lexer
  // needs to know if we are inside one
  bool IsBlockComment = false;
  struct BlockCommentRaii {
    Lexer& L;
    bool Prev;
    BlockCommentRaii(Lexer& L)
      : L(L), Prev(L.IsBlockComment)
    { L.IsBlockComment = true; }
    ~BlockCommentRaii() { L.IsBlockComment = Prev; }

    // This should cancel reverting the IsBlockComment
    void setInvalidEof() {
      Prev = true;
    }
  };

  void LexIdentifier(Token& Tok, const char *CurPtr);
  void LexNumberOrIdentifier(Token& Tok, const char *CurPtr);
  void LexNumberOrEllipsis(Token& Tok, const char *CurPtr);
  void LexNumber(Token& Tok, const char *CurPtr);
  void LexSharpLiteral(Token& Tok, const char *CurPtr);
  void LexStringLiteral(Token& Tok, const char *CurPtr);
  void LexUnknown(Token& Tok, const char *CurPtr);
  void SkipUntilDelimiter(const char *&CurPtr);
  void ProcessWhitespace(const char *&CurPtr);
  void ProcessBlockComment(const char *&CurPtr);
  bool TryProcessComment(const char *&CurPtr);

  // Advances the Ptr and returns the char
  char ConsumeChar(const char *&Ptr) {
    return *(++Ptr);
  }

  void FormIdentifier(Token &Result, const char *TokEnd) {
    FormTokenWithChars(Result, TokEnd, tok::identifier);
  }

  // TODO deprecate this
  void FormLiteral(Token &Result, const char *TokEnd, TokenKind Kind) {
    FormTokenWithChars(Result, TokEnd, Kind);
  }

  // Copy/Pasted from Lexer
  void FormTokenWithChars(Token &Result, const char *TokEnd,
                          TokenKind Kind) {
    unsigned TokLen = TokEnd - BufferPtr;
    Result.Kind = Kind;
    Result.LiteralData = llvm::StringRef(BufferPtr, TokLen);
    Result.Loc = getSourceLocation(BufferPtr);
    BufferPtr = TokEnd;
  }
  SourceLocation getSourceLocation(const char *Loc) const;

public:
  Lexer() = default;

  Lexer(SourceFile File)
    : EmbeddedLexer(File.StartLoc, File.Buffer, File.Buffer.begin())
  { }

  Lexer(SourceFile File, char const* BufferPos)
    : EmbeddedLexer(File.StartLoc, File.Buffer, BufferPos)
  { }

  void Lex(Token& Tok);
};

} // namespace heavy

#endif
