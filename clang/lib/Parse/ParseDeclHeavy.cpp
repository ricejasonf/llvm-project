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

#include "clang/Parse/Parser.h"
#include "clang/Parse/ParserHeavyScheme.h"
#include "clang/AST/HeavyScheme.h"
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
heavy::Value* LoadEmbeddedEnv(heavy::Context& Context,
                              DeclContext* DC) {
  auto itr = Context.EmbeddedEnvs.find(DC);
  if (itr != Context.EmbeddedEnvs.end()) return itr->second;
  Value* Env;
  if (DC->isTranslationUnit()) {
    Env = Context.SystemEnvironment;
  } else {
    Env = LoadEmbeddedEnv(DC->getParent());
  }
  Env = Context.CreatePair(Context.CreateModule(), Env);
  Context.EmbeddedEnvs[DC] = Env;
  return Env;
}

} // end anon namespace

bool Parser::ParseHeavyScheme() {
  if (!HeavySchemeContext) {
    HeavySchemeContext = heavy::Context::CreateEmbedded(*this);
  }
  heavy::Context& Context = *HeavySchemeContext;
  auto SchemeLexer = HeavySchemeLexer();
  PP.InitEmbeddedLexer(SchemeLexer);

  ParserHeavyScheme P(SchemeLexer, Context, *this);

  P.ConsumeToken();
  if (!P.TryConsumeToken(tok::l_brace)) {
    Diag(Tok, diag::err_expected) << tok::l_brace;
    return true;
  }

  // Load the environment for the current DeclContext
  DeclContext* DC = getActions().CurContext;
  Context.EnvStack = P.LoadEmbeddedEnv(DC);

  heavy::ValueResult Result;
  bool HasError = Context.CheckError();
  while (true) {
    if (!HasError && Context.CheckError()) {
      HasError = true;
      Diag(Context.getErrorLocation(), diag::err_heavy_scheme)
        << Context.getErrorMessage();
    }
    // Keep parsing until we find the end
    // brace (represented by isUnset() here)
    Result = P.ParseTopLevelExpr();
    if (Result.isUnset()) break;
    if (HasError) continue;
    if (Result.isUsable()) {
      heavy::Value* Val = eval(Context, Result.get());
      // TODO just discard the value without dumping it
      if (!Context.CheckError()) Val->dump();
    }
  };

  // Return control to C++ Lexer
  PP.FinishEmbeddedLexer(SchemeLexer);

  // Allow subsequent heavy_scheme
  // instances to render useful errors
  // as well
  Context.ClearError();


  // The Lexers position has been changed
  // so we need to re-prime the look-ahead
  this->ConsumeToken();

  return HasError;
}
