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
#include "clang/AST/Expr.h"
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
heavy::ExternLambda<1> HEAVY_CLANG_VAR(write_lexer) = {};
heavy::ExternLambda<1, sizeof(void*)*2> HEAVY_CLANG_VAR(expr_eval) = {};

namespace {
// Convert to a clang::SourceLocation or an invalid location if it
// is not external.
clang::SourceLocation getSourceLocation(heavy::FullSourceLocation Loc) {
  if (!Loc.isExternal()) return clang::SourceLocation();
  return clang::SourceLocation
    ::getFromRawEncoding(Loc.getExternalRawEncoding())
     .getLocWithOffset(Loc.getOffset());
}

void LoadParentEnv(heavy::HeavyScheme& HS, void* Handle) {
  DeclContext* DC = reinterpret_cast<DeclContext*>(Handle);
  if (!DC->isTranslationUnit()) {
    HS.LoadEmbeddedEnv(DC->getParent(), LoadParentEnv);
  }
}

// It is complicated to keep the TokenBuffer alive
// for the Preprocessor, so we use an array to give
// ownership via the EnterTokenStream overload.
class LexerWriter {
  clang::Parser& Parser;
  heavy::HeavyScheme& HeavyScheme;
  std::unique_ptr<Token[]> TokenBuffer;
  unsigned Capacity = 0;
  unsigned Size = 0;

  void realloc(unsigned NewCapacity) {
    std::unique_ptr<Token[]> NewTokenBuffer(new Token[NewCapacity]());
    std::copy(&TokenBuffer[0], &TokenBuffer[Size],
              NewTokenBuffer.get());
    TokenBuffer = std::move(NewTokenBuffer);
    Capacity = NewCapacity;
  }

  void push_back(Token Tok) {
    unsigned NewSize = Size + 1;
    if (Capacity < NewSize) {
      // Start with a reasonable 128 bytes and then
      // double capacity each time it is needed.
      unsigned NewCapacity = Capacity > 0 ? Capacity * 2 : 128;
      realloc(NewCapacity);
    }
    TokenBuffer[Size] = Tok;
    Size = NewSize;
  }

public:
  LexerWriter(clang::Parser& P, heavy::HeavyScheme& HS)
    : Parser(P),
      HeavyScheme(HS),
      TokenBuffer(nullptr)
  { }

  ~LexerWriter() {
    Capacity = 0;
    Size = 0;
  }

  void operator()(heavy::Context& C, heavy::ValueRefs Args) {
    if (Args.size() != 1) return C.RaiseError("invalid arity");
    if (!isa<heavy::String>(Args[0]))
      return C.RaiseError("expecting string");
    llvm::StringRef Result = cast<heavy::String>(Args[0])->getView();
    // Lex Tokens for the TokenBuffer.
    clang::Lexer Lexer(clang::SourceLocation(), Parser.getLangOpts(),
                       Result.data(), Result.data(), &(*(Result.end())));
    while (true) {
      Token Tok;
      Lexer.LexFromRawLexer(Tok);

      // Raw identifiers need to be looked up.
      if (Tok.is(tok::raw_identifier))
        Parser.getPreprocessor().LookUpIdentifierInfo(Tok);


      Tok.setLocation(getSourceLocation(
          HeavyScheme.getFullSourceLocation(C.getLoc())));

      if (Tok.is(tok::eof)) break;
      push_back(Tok);
    }

    C.Cont();
  }

  // This must be called AFTER we update the Clang Lexer position.
  void FlushTokens() {
    if (Size == 0) return;
    Parser.getPreprocessor().EnterTokenStream(std::move(TokenBuffer), Size,
                    /*DisableMacroExpansion=*/true,
                    /*IsReinject=*/false);
    Capacity = 0;
    Size = 0;
  }
};

void LoadBuiltinModule(clang::Parser& P, heavy::HeavyScheme* HS) {
  // FIXME We are capturing Parser and storing it in globals.
  //       This is a problem if we have multiple CompilerInvocations.
  //       There must be a way to store "globals" with Context if
  //       they have state.
  //       (This is a problem for static modules in general.)
  auto diag_error = [&](heavy::Context& C, heavy::ValueRefs Args) {
    if (Args.size() != 1) {
      C.RaiseError("invalid arity to function", C.getCallee());
      return;
    }
    if (!isa<heavy::String>(Args[0])) {
      C.RaiseError("expecting string", C.getCallee());
      return;
    }
    llvm::StringRef Err = cast<heavy::String>(Args[0])->getView();

    P.Diag(clang::SourceLocation{}, diag::err_heavy_scheme) << Err;
    C.Cont();
  };

  auto hello_world = [](heavy::Context& C, heavy::ValueRefs Args) {
    llvm::errs() << "hello world (from clang)\n";
    C.Cont();
  };

  auto expr_eval = [&](heavy::Context& C, heavy::ValueRefs Args) {
    if (Args.size() != 1) {
      C.RaiseError("invalid arity to function", C.getCallee());
      return;
    }
    if (!isa<heavy::String>(Args[0])) {
      C.RaiseError("expecting string", C.getCallee());
      return;
    }
    llvm::StringRef Source = cast<heavy::String>(Args[0])->getView();


    // Lex and expand.
    clang::Lexer Lexer(clang::SourceLocation(), P.getLangOpts(),
                       Source.data(), Source.data(), &(*(Source.end())));
    LexerWriter TheLexerWriter(P, *HS);
    TheLexerWriter.FlushTokens();

    // Parse the expression.
    clang::ExprResult ExprResult = P.ParseExpression();

    // TODO
    // Make sure the macro expansion is popped off the include stack.

    // Process the parsing result if any.
    if (ExprResult.isInvalid()) {
      return C.RaiseError("clang expression parsing failed");
    }
    clang::Expr* Expr = ExprResult.get();

    // ConstantExpr eval.
    clang::Expr::EvalResult EvalResult;
    if (!Expr->EvaluateAsRValue(EvalResult,
          P.getActions().getASTContext())) {
      // The evaluation failed.
      // TODO Have Clang emit the diagnostics (ie From EvalResult.Diag)
      C.RaiseError("clang expression evaluation failed");
      return;
    }

    // Convert EvalResult/APValue to Scheme value.
    heavy::Value Result;
    switch (EvalResult.Val.getKind()) {
      case APValue::None:
      case APValue::Indeterminate: {
        // ... Or maybe we allow errors. (ie like SFINAE)
        Result = heavy::Undefined();
        C.RaiseError("clang expression evaluation failed");
        return;
      }
      case APValue::Int: {
        llvm::APSInt Int = EvalResult.Val.getInt();
        if (Expr->getType()->isBooleanType()) {
          Result = heavy::Bool(Int.getBoolValue());
        } else if (Int.isSignedIntN(32)) {
          Result = heavy::Int(Int.getZExtValue());
        }
        break;
      }
      // TODO Support these value types.
      case APValue::Float:
      case APValue::FixedPoint:
      case APValue::ComplexInt:
      case APValue::ComplexFloat:
      case APValue::Vector:
      case APValue::Array:
      case APValue::Struct:

      // Will not support.
      case APValue::LValue:
      case APValue::Union:
      case APValue::MemberPointer:
      case APValue::AddrLabelDiff:
      default:
        // Do nothing.
      break;
    }

    if (Result) {
      C.Cont(Result);
    } else {
      C.RaiseError("unsupported result type");
    }
  };

  HEAVY_CLANG_VAR(diag_error)   = diag_error;
  HEAVY_CLANG_VAR(hello_world)  = hello_world;
  HEAVY_CLANG_VAR(expr_eval)    = expr_eval;
}
} // end anon namespace

bool Parser::ParseHeavyScheme() {
  if (!HeavyScheme) {
    HeavyScheme = std::make_unique<heavy::HeavyScheme>();
    LoadBuiltinModule(*this, HeavyScheme.get());
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
    clang::SourceLocation ErrLoc = getSourceLocation(EmbeddedLoc);
    Diag(ErrLoc, diag::err_heavy_scheme) << Err;
  };


  LexerWriter TheLexerWriter(*this, *HeavyScheme);
  HEAVY_CLANG_VAR(write_lexer) = [&](heavy::Context& C,
                                     heavy::ValueRefs Args) mutable {
    TheLexerWriter.operator()(C, Args);
  };
  heavy::TokenKind Terminator = heavy::tok::r_brace;
  HeavyScheme->ProcessTopLevelCommands(SchemeLexer,
                                       ErrorHandler,
                                       Terminator);

  // Return control to C++ Lexer
  PP.FinishEmbeddedLexer(SchemeLexer.GetByteOffset());
  TheLexerWriter.FlushTokens();
  HEAVY_CLANG_VAR(write_lexer) = [&](heavy::Context& C,
                                     heavy::ValueRefs Args) mutable {
    C.RaiseError("lexer writer is not initialized");
  };

  // The Lexers position has been changed
  // so we need to re-prime the look-ahead
  this->ConsumeToken();

  return HasError;
}
