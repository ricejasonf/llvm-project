// Copyright Jason Rice 2025

#include <heavy/Builtins.h>
#include <heavy/Clang.h>
#include <heavy/Context.h>
#include <heavy/HeavyScheme.h>
#include <heavy/Value.h>
#include <clang/AST/Expr.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/Parser.h>
#include <utility>

heavy::ContextLocal HEAVY_CLANG_VAR(diag_error);
heavy::ContextLocal HEAVY_CLANG_VAR(diag_warning);
heavy::ContextLocal HEAVY_CLANG_VAR(diag_note);
heavy::ContextLocal HEAVY_CLANG_VAR(hello_world);
heavy::ContextLocal HEAVY_CLANG_VAR(write_lexer);
heavy::ContextLocal HEAVY_CLANG_VAR(lexer_writer);
heavy::ContextLocal HEAVY_CLANG_VAR(expr_eval);

namespace {
using Pair = std::pair<clang::Parser*, std::unique_ptr<heavy::HeavyScheme>>;

static auto Instance = Pair();

// Convert to a clang::SourceLocation or an invalid location if it
// is not external.
clang::SourceLocation getSourceLocation(heavy::FullSourceLocation Loc) {
  if (!Loc.isExternal()) return clang::SourceLocation();
  return clang::SourceLocation
    ::getFromRawEncoding(Loc.getExternalRawEncoding())
     .getLocWithOffset(Loc.getOffset());
}

template <clang::DiagnosticsEngine::Level Level>
struct DiagReport {
  void operator()(heavy::HeavyScheme& HS,
                  heavy::SourceLocation Loc,
                  clang::DiagnosticsEngine& Diags,
                  llvm::StringRef ErrMsg) const {
    heavy::FullSourceLocation FullLoc = HS.getFullSourceLocation(Loc);
    this->operator()(HS, FullLoc, Diags, ErrMsg);
  }

  void operator()(heavy::HeavyScheme& HS,
                  heavy::FullSourceLocation HSLoc,
                  clang::DiagnosticsEngine& Diags,
                  llvm::StringRef ErrMsg) const {
    // Create a custom DiagId once for our instance.
    static heavy::ContextLocal CustomDiagId;
    heavy::Context& Context = HS.getContext();
    heavy::Binding* DiagIdBinding = CustomDiagId.getBinding(Context);
    if (heavy::isa<heavy::Undefined>(DiagIdBinding->getValue())) {
      unsigned Id = Diags.getCustomDiagID(Level, "(heavy_scheme) %0");
      DiagIdBinding->setValue(heavy::Int(static_cast<int32_t>(Id)));
    }
    unsigned DiagId = static_cast<unsigned>(
        heavy::cast<heavy::Int>(DiagIdBinding->getValue()));
    clang::SourceLocation Loc = getSourceLocation(HSLoc);
    Diags.Report(Loc, DiagId) << ErrMsg;
  }
};

// It is complicated to keep the TokenBuffer alive
// for the Preprocessor, so we use an array to give
// ownership via the EnterTokenStream overload.
class LexerWriter {
  using Token = clang::Token;
  clang::Parser& Parser;
  llvm::BumpPtrAllocator& LexerSpellings;
  std::unique_ptr<Token[]> TokenBuffer;
  unsigned Capacity = 0;
  unsigned Size = 0;

  void realloc(unsigned NewCapacity) {
    std::unique_ptr<Token[]> NewTokenBuffer(new Token[NewCapacity]());
    if (Capacity > 0)
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
  LexerWriter(clang::Parser& P,
              llvm::BumpPtrAllocator& LexerSpellings)
    : Parser(P),
      LexerSpellings(LexerSpellings),
      TokenBuffer(nullptr)
  { }

  ~LexerWriter() {
    Capacity = 0;
    Size = 0;
  }

  // Lex tokens from string and push to TokenBuffer.
  // Copy to a std::string to guarantee a null terminator.
  void Tokenize(clang::SourceLocation Loc, llvm::StringRef Chars) {
    if (Chars.empty()) return;
    // Copy to LexerSpellings to ensure null terminator.
    Chars = Chars.copy(LexerSpellings);
    char* NullTerm = LexerSpellings.template Allocate<char>(1);
    *NullTerm = 0;
    // Lex Tokens for the TokenBuffer.
    clang::Lexer Lexer(clang::SourceLocation(), Parser.getLangOpts(),
            Chars.data(), Chars.data(), &(*(Chars.end())));
    while (true) {
      Token Tok;
      Lexer.LexFromRawLexer(Tok);

      // Raw identifiers need to be looked up.
      if (Tok.is(clang::tok::raw_identifier))
        Parser.getPreprocessor().LookUpIdentifierInfo(Tok);

      Tok.setLocation(Loc);

      if (Tok.is(clang::tok::eof)) break;
      push_back(Tok);
    }
  }

  // This must be called AFTER we update the Clang Lexer position.
  void FlushTokens() {
    if (Size == 0) return;
    Parser.getPreprocessor().EnterTokenStream(std::move(TokenBuffer), Size,
                    /*DisableMacroExpansion=*/true,
                    /*IsReinject=*/true);
    Capacity = 0;
    Size = 0;
  }
};


class HeavySchemePragmaHandler : public clang::ParserPragmaHandler {
public:
  HeavySchemePragmaHandler()
    : ParserPragmaHandler("heavy_scheme")
  { }

  void CreateInst(clang::Parser& P, Pair& Inst) {
    auto& [ParserPtr, HeavyScheme] = Inst;
    ParserPtr = &P;
    HeavyScheme = std::make_unique<heavy::HeavyScheme>();
    HeavyScheme->LexerSpellings = std::make_unique<llvm::BumpPtrAllocator>();
    // Load the static builtin module.
    heavy::HeavyScheme& HS = *HeavyScheme;
    auto diag_gen = [&](auto DiagReportFn) {
      return [&, DiagReportFn](heavy::Context& C, heavy::ValueRefs Args) {
        if (Args.size() < 1 || Args.size() > 2) {
          C.RaiseError("invalid arity to function", C.getCallee());
          return;
        }

        if (!heavy::isa<heavy::String, heavy::Symbol>(Args[0])) {
          C.RaiseError("expecting string or identifier", C.getCallee());
          return;
        }

        llvm::StringRef Err = Args[0].getStringRef();
        heavy::SourceLocation Loc;
        if (Args.size() > 1)
          Loc = Args[1].getSourceLocation();

        auto& Diags = P.getPreprocessor().getDiagnostics();
        DiagReportFn(HS, Loc, Diags, Err);
        C.Cont();
      };
    };

    auto diag_error = diag_gen(
        DiagReport<clang::DiagnosticsEngine::Level::Error>{});
    auto diag_warning = diag_gen(
        DiagReport<clang::DiagnosticsEngine::Level::Warning>{});
    auto diag_note = diag_gen(
        DiagReport<clang::DiagnosticsEngine::Level::Note>{});

    auto hello_world = [](heavy::Context& C, heavy::ValueRefs Args) {
      llvm::errs() << "hello world (from clang)\n";
      C.Cont();
    };

    auto expr_eval = [&](heavy::Context& C, heavy::ValueRefs Args) {
      heavy::SourceLocation Loc;
      heavy::Value Input;
      if (Args.size() == 2) {
        // Accept any value that may have a source location.
        Loc = Args[0].getSourceLocation();
        Input = Args[1];
      } else if (Args.size() == 1) {
        Input = Args[0];
      } else {
        return C.RaiseError("invalid arity");
      }
      if (!heavy::isa<heavy::String, heavy::Symbol>(Input)) {
        C.RaiseError("expecting string or identifier", C.getCallee());
        return;
      }
      llvm::StringRef Source = Input.getStringRef();
      if (!Loc.isValid())
        Loc = Input.getSourceLocation();
      if (!Loc.isValid())
        Loc = C.getLoc();

      // FIXME Check to see if we ever need this TentativeParsingAction.
      // Prepare to revert Parser.
      //clang::Parser::TempExposedTentativeParsingAction ParseReverter(P);

      // Lex and expand.
      LexerWriter TheLexerWriter(P, *HS.LexerSpellings);
      TheLexerWriter.Tokenize(getSourceLocation(HS.getFullSourceLocation(Loc)),
                              Source);
      TheLexerWriter.FlushTokens();
      P.ConsumeAnyToken();

      // Parse the expression.
      clang::ExprResult ExprResult = P.ParseExpression();

      // Revert the lexer position so we don't keep moving forward.
      //ParseReverter.Revert();

      // Process the parsing result if any.
      if (ExprResult.isInvalid()) {
        return C.RaiseError("clang expression parsing failed");
      }
      clang::Expr* Expr = ExprResult.get();

      if (Expr->isValueDependent()) {
        return C.RaiseError("cannot evaluate dependent expression");
      }

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
      using APValue = clang::APValue;
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
          // Do nothing.
        break;
      }

      if (Result) {
        C.Cont(Result);
      } else {
        C.RaiseError("unsupported result type");
      }
    };

    // This is a special system specific function so we can
    // use Clang's file search and source locations.
    auto ParseSourceFileFn = [&P, &HS](heavy::Context& C,
                                    heavy::SourceLocation Loc,
                                    heavy::String* RequestedFilename) {
      clang::Preprocessor& PP = P.getPreprocessor();
      heavy::FullSourceLocation
        FullLoc = HS.getFullSourceLocation(Loc);
      clang::SourceLocation ClangLoc = getSourceLocation(FullLoc);
      clang::OptionalFileEntryRef File = PP.LookupFile(
          ClangLoc, RequestedFilename->getView(),
          false, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr);
      if (!File) {
        heavy::String* ErrMsg = C.CreateString("error opening file: ",
                                            RequestedFilename->getStringRef());
        return C.RaiseError(ErrMsg, heavy::Value(RequestedFilename));
      }
      // Determine if file is a system file... as if!
      clang::SrcMgr::CharacteristicKind FileChar = 
        PP.getHeaderSearchInfo()
          .getFileDirFlavor(*File);
      clang::FileID FileId = 
        PP.getSourceManager().createFileID(*File, ClangLoc, FileChar);
      clang::SourceLocation StartLoc =
        PP.getSourceManager().getLocForStartOfFile(FileId);
      std::optional<llvm::MemoryBufferRef> Buffer =
        PP.getSourceManager().getBufferOrNone(FileId, ClangLoc);
      if (!Buffer)
        return C.RaiseError("error opening file buffer",
                            heavy::Value(RequestedFilename));
      // Is it over yet?
      C.Cont(HS.ParseSourceFile(StartLoc.getRawEncoding(),
                                File->getName(),
                                Buffer->getBufferStart(),
                                Buffer->getBufferEnd(),
                                Buffer->getBufferStart()));
    };

    heavy::Context& Context = HeavyScheme->getContext();
    heavy::builtins::InitParseSourceFile(Context, ParseSourceFileFn);
    HEAVY_CLANG_VAR(diag_error).set(Context,
                                    Context.CreateLambda(diag_error));
    HEAVY_CLANG_VAR(diag_warning).set(Context,
                                      Context.CreateLambda(diag_warning));
    HEAVY_CLANG_VAR(diag_note).set(Context,
                                   Context.CreateLambda(diag_note));
    HEAVY_CLANG_VAR(hello_world).set(Context,
                                     Context.CreateLambda(hello_world));
    HEAVY_CLANG_VAR(expr_eval).set(Context,
                                    Context.CreateLambda(expr_eval));
    HeavyScheme->RegisterModule(HEAVY_CLANG_LIB_STR, HEAVY_CLANG_LOAD_MODULE);
  }

  void HandleParseExternalDeclaration(
                    clang::Parser& P,
                    clang::Token& Tok,
                    clang::DeclGroupRef&) override {
    auto& [ParserPtr, HeavyScheme] = Instance;
    // Only supporting one instance.
    if (ParserPtr == nullptr)
      CreateInst(P, Instance);
    else if (&P != ParserPtr)
      return;

    heavy::Lexer SchemeLexer;
    auto LexerInitFn = [&](clang::SourceLocation Loc,
                           llvm::StringRef Name,
                           char const* BufferStart,
                           char const* BufferEnd,
                           char const* BufferPtr) {
      SchemeLexer = HeavyScheme->createEmbeddedLexer(
                          Loc.getRawEncoding(), Name,
                          BufferStart, BufferEnd, BufferPtr);
    };

    clang::Preprocessor& PP = P.getPreprocessor();
    PP.InitEmbeddedLexer(LexerInitFn);

    bool HasError = false;
    auto ErrorHandler = [&](llvm::StringRef Err,
                            heavy::FullSourceLocation EmbeddedLoc) {
      HasError = true;
      auto& Diags = PP.getDiagnostics();
      DiagReport<clang::DiagnosticsEngine::Level::Error>{}(
            *HeavyScheme, EmbeddedLoc, Diags, Err);
    };

    LexerWriter TheLexerWriter(P, *HeavyScheme->LexerSpellings);
    heavy::Context& Context = HeavyScheme->getContext();
    HEAVY_CLANG_VAR(write_lexer).set(Context, 
        Context.CreateLambda([&](heavy::Context& C,
                                 heavy::ValueRefs Args) mutable {
      heavy::SourceLocation Loc;
      heavy::Value Output;
      if (Args.size() == 2) {
        // Accept any value that may have a source location.
        Loc = Args[0].getSourceLocation();
        Output = Args[1];
      } else if (Args.size() == 1) {
        Output = Args[0];
      } else {
        return C.RaiseError("invalid arity");
      }
      if (!heavy::isa<heavy::String, heavy::Symbol>(Output))
        return C.RaiseError(C.CreateString(
              "invalid type: ",
              heavy::getKindName(Output.getKind()),
              ", expecting string or identifier"
              ), Output);

      // Try to get a valid source location.
      if (!Loc.isValid()) Loc = Output.getSourceLocation();
      if (!Loc.isValid()) Loc = C.getLoc();

      llvm::StringRef Result = Output.getStringRef();
      TheLexerWriter.Tokenize(getSourceLocation(
            HeavyScheme->getFullSourceLocation(Loc)),
            Result);
      C.Cont();
    }));

    // Also provide a type erased LexerWriterFnRef which is
    // more suited to calling in c++.
    auto LexerWriterFn = [&](heavy::SourceLocation Loc, llvm::StringRef Str) {
      TheLexerWriter.Tokenize(getSourceLocation(
            HeavyScheme->getFullSourceLocation(Loc)), Str);
    };
    auto LWF = heavy::LexerWriterFnRef(LexerWriterFn);
    HEAVY_CLANG_VAR(lexer_writer).set(Context, Context.CreateAny(LWF));

    // Do the thing.
    heavy::TokenKind Terminator = heavy::tok::r_brace;
    HeavyScheme->ProcessTopLevelCommands(SchemeLexer, heavy::builtins::eval,
                                         ErrorHandler, Terminator);

    // Return control to C++ Lexer
    PP.FinishEmbeddedLexer(SchemeLexer.GetByteOffset());
    if (!HasError)
      TheLexerWriter.FlushTokens();
    HEAVY_CLANG_VAR(lexer_writer).set(Context, heavy::Undefined());
    HEAVY_CLANG_VAR(write_lexer).set(Context,
          Context.CreateBuiltin([](heavy::Context& C, heavy::ValueRefs Args) {
      C.RaiseError("lexer writer is not initialized");
    }));

    // The Lexers position has been changed
    // so we need to re-prime the look-ahead
    P.ConsumeAnyToken();
  }
};

} // namespace

static clang::PragmaHandlerRegistry::Add<HeavySchemePragmaHandler>
PragmaHandler("heavy-scheme", "embed compile-time scheme");
