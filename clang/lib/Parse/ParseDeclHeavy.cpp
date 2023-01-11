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

#include "heavy/Clang.h"
#include "heavy/Context.h"
#include "heavy/HeavyScheme.h"
#include "heavy/Value.h"
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

bool HEAVY_CLANG_IS_LOADED = false;
heavy::ExternLambda<1> HEAVY_CLANG_VAR(diag_error) = {};
heavy::ExternLambda<1> HEAVY_CLANG_VAR(hello_world) = {};

namespace {
void LoadParentEnv(heavy::HeavyScheme& HS, void* Handle) {
  DeclContext* DC = reinterpret_cast<DeclContext*>(Handle);
  if (!DC->isTranslationUnit()) {
    HS.LoadEmbeddedEnv(DC->getParent(), LoadParentEnv);
  }
}

void LoadBuiltinModule(clang::Parser& P) {
  auto diag_error = [&](heavy::Context& C, heavy::ValueRefs Args) {
    // TODO automate argument checking somehow
    if (Args.size() != 1) {
      C.RaiseError("invalid arity to function", C.getCallee());
      return;
    }
    if (!isa<heavy::String>(Args[0])) {
      C.RaiseError("expecting string", C.getCallee());
      return;
    }
    llvm::StringRef Err = cast<heavy::String>(Args[0])->getView();

    P.Diag(clang::SourceLocation{}, diag::err_heavy_scheme)
      << "MESSAGE FROM CLANG LAND: " << Err;
    C.Cont(); 
  };

  auto hello_world = [](heavy::Context& C, heavy::ValueRefs Args) {
    llvm::errs() << "\nhello world (from clang)\n";
    C.Cont(); 
  };

  HEAVY_CLANG_VAR(diag_error)   = diag_error;
  HEAVY_CLANG_VAR(hello_world)  = hello_world;
}
} // end anon namespace

bool Parser::ParseHeavyScheme() {
  if (!HeavyScheme) {
    HeavyScheme = std::make_unique<heavy::HeavyScheme>();
    LoadBuiltinModule(*this);
    HeavyScheme->RegisterModule(HEAVY_CLANG_LIB_STR, HEAVY_CLANG_LOAD_MODULE);
  }

  heavy::Lexer SchemeLexer;
  auto LexerInitFn = [&](clang::SourceLocation Loc,
                         char const* BufferStart,
                         char const* BufferEnd,
                         char const* BufferPtr) {
    SchemeLexer = HeavyScheme->createEmbeddedLexer(
                        Loc.getRawEncoding(),
                        BufferStart,
                        BufferEnd,
                        BufferPtr);
  };

  PP.InitEmbeddedLexer(LexerInitFn);

  // Load the environment for the current DeclContext
  DeclContext* DC = getActions().CurContext;
  HeavyScheme->LoadEmbeddedEnv(DC, LoadParentEnv);

  bool HasError = false;
  auto ErrorHandler = [&](llvm::StringRef Err,
                          heavy::FullSourceLocation EmbeddedLoc) {
    HasError = true;
    clang::SourceLocation ErrLoc = clang::SourceLocation
      ::getFromRawEncoding(EmbeddedLoc.getExternalRawEncoding())
       .getLocWithOffset(EmbeddedLoc.getOffset());
    Diag(ErrLoc, diag::err_heavy_scheme) << Err;
  };


  heavy::TokenKind Terminator = heavy::tok::r_brace;
  HeavyScheme->ProcessTopLevelCommands(SchemeLexer,
                                       ErrorHandler,
                                       Terminator);

  // Return control to C++ Lexer
  PP.FinishEmbeddedLexer(SchemeLexer.GetByteOffset());

  // The Lexers position has been changed
  // so we need to re-prime the look-ahead
  this->ConsumeToken();

  return HasError;
}
