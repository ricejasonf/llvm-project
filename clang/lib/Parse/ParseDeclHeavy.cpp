//===--- ParseDeclHeavy.cpp - HeavyScheme Declaration Parsing ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Heavy Declaration portions of the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "heavy/HeavyScheme.h"
#include "clang/Parse/Parser.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyDeclStackTrace.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;

namespace {
void LoadParentEnv(heavy::HeavyScheme& HS, void* Handle) {
  DeclContext* DC = reinterpret_cast<DeclContext*>(Handle);
  if (DC->isTranslationUnit()) {
    HS.LoadCoreEnv();
  } else {
    HS.LoadEmbeddedEnv(DC->getParent(), LoadParentEnv);
  }
}
} // end anon namespace

bool Parser::ParseHeavyScheme() {
  HeavyScheme.init();
  heavy::Lexer SchemeLexer;
  auto LexerInitFn = [&](clang::SourceLocation Loc,
                         char const* BufferStart,
                         char const* BufferEnd,
                         char const* BufferPtr) {
    SchemeLexer = HeavyScheme.createEmbeddedLexer(
                        Loc.getRawEncoding(),
                        BufferStart,
                        BufferEnd,
                        BufferPtr);
  };

  PP.InitEmbeddedLexer(LexerInitFn);

  // Load the environment for the current DeclContext
  DeclContext* DC = getActions().CurContext;
  HeavyScheme.LoadEmbeddedEnv(DC, LoadParentEnv);

  heavy::TokenKind Terminator = heavy::tok::r_brace;
  bool HasError = HeavyScheme.ProcessTopLevelCommands(SchemeLexer,
                                                      Terminator);

  // Return control to C++ Lexer
  PP.FinishEmbeddedLexer(SchemeLexer.GetByteOffset());

  // The Lexers position has been changed
  // so we need to re-prime the look-ahead
  this->ConsumeToken();

  return HasError;
}
