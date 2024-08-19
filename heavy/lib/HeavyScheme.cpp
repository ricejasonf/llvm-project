//===--- HeavyScheme.cpp - HeavyScheme Context Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for the opaque interfaces in HeavyScheme.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/Context.h"
#include "heavy/HeavyScheme.h"
#include "heavy/Lexer.h"
#include "heavy/Mangle.h"
#include "heavy/OpGen.h"
#include "heavy/Parser.h"
#include "heavy/Source.h"
#include "heavy/SourceManager.h"
#include "heavy/Value.h"


namespace heavy {

HeavyScheme::HeavyScheme(std::unique_ptr<heavy::Context> C)
  : ContextPtr(std::move(C)),
    EnvPtr(std::make_unique<heavy::Environment>(*ContextPtr)),
    SourceManagerPtr(std::make_unique<heavy::SourceManager>()),
    SourceFileStoragePtr(nullptr, [](SourceFileStorage*) { })
{
  ContextPtr->setEnvironment(EnvPtr.get());
  HEAVY_BASE_INIT(*ContextPtr);
}

HeavyScheme::HeavyScheme()
  : HeavyScheme(std::make_unique<heavy::Context>())
{ }

HeavyScheme::~HeavyScheme() = default;

// Create a Lexer using a "file" from some externally managed source.
heavy::Lexer HeavyScheme::createEmbeddedLexer(uintptr_t ExternalRawLoc,
                                              llvm::StringRef Name,
                                              char const* BufferStart,
                                              char const* BufferEnd,
                                              char const* BufferPos) {
  heavy::SourceManager& SM = getSourceManager();
  size_t BufferLen = BufferEnd - BufferStart;
  llvm::StringRef FileBuffer(BufferStart, BufferLen);
  heavy::SourceFile File = SM.getOrCreateExternal(ExternalRawLoc, FileBuffer,
                                                  Name);
  return Lexer(File, BufferPos);
}

void HeavyScheme::LoadEmbeddedEnv(void* Handle,
          llvm::function_ref<void(HeavyScheme&, void*)> LoadParent) {
  auto& Context = getContext();
  auto& EmbeddedEnvs = Context.EmbeddedEnvs; // TODO store EmbeddedEnvs in HeavyScheme
  auto itr = EmbeddedEnvs.find(Handle);
  if (itr != EmbeddedEnvs.end()) {
    Context.EnvStack = itr->second.get();
    return;
  }

  LoadParent(*this, Handle);
  Environment* Parent = cast<Environment>(Context.getEnvironment());
  auto& Env = Context.EmbeddedEnvs[Handle] =
    std::make_unique<Environment>(Parent);
  Context.setEnvironment(Env.get());
  return;
}

void HeavyScheme::ProcessTopLevelCommands(
                              heavy::Lexer& Lexer,
                              llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                              heavy::tok Terminator) {
  return ProcessTopLevelCommands(Lexer, base::eval, ErrorHandler,
                                 Terminator);
}

void HeavyScheme::ProcessTopLevelCommands(
                              heavy::Lexer& Lexer,
                              llvm::function_ref<ValueFnTy> ExprHandler,
                              llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                              heavy::tok Terminator) {
  // This should be the only entry point so that Env is captured and
  // participates in garbage collection.
  heavy::Environment* Env = EnvPtr.get();
  auto& Context = getContext();
  auto& SM = getSourceManager();
  auto ParserPtr = std::make_unique<Parser>(Lexer, Context);
  Parser& Parser = *ParserPtr;

  auto HandleErrorFn = [&](heavy::Context& Context, ValueRefs Args) {
    heavy::FullSourceLocation FullLoc = SM.getFullSourceLocation(
        Args[0].getSourceLocation());
    if (heavy::Error* Err = dyn_cast<heavy::Error>(Args[0])) {
      ErrorHandler(Err->getErrorMessage(), FullLoc);
    } else {
      ErrorHandler("errorhandler received a non-error", FullLoc);
    }
    // Finish parsing to return control to any
    // parent lexer. (ie So c++ does not parse scheme code)
    while (!Parser.isFinished()) {
      Parser.ParseTopLevelExpr();
    }
    Context.Cont();
  };

  if (!Parser.PrimeToken(Terminator)) {
    HandleErrorFn(Context, {});
    return;
  }

  Value HandleExpr = Context.CreateLambda(ExprHandler, {});
  Value HandleError = Context.CreateLambda(HandleErrorFn, {});
  Value MainThunk = Context.CreateLambda([&Parser]
                          (heavy::Context& C, ValueRefs) {
    if (Parser.isFinished()) {
      C.Cont();
      return;
    }

    // Recurse in tail position.
    C.PushCont(C.getCallee());

    heavy::ValueResult ParseResult = Parser.ParseTopLevelExpr();
    if (ParseResult.isUsable()) {
      Value Env = C.getCapture(0);
      Value HandleExpr = C.getCapture(1);
      Value ParseResultVal = ParseResult.get();
      C.setLoc(ParseResultVal.getSourceLocation());
      std::array<Value, 2> EvalArgs{ParseResultVal, Env};
      C.Apply(HandleExpr, EvalArgs);
    } else if (Parser.HasError()) {
      Parser.RaiseError();
    } else {
      C.Cont();
    }
  }, CaptureList{Value(Env), HandleExpr});

  Context.DynamicWind(std::move(ParserPtr), Context.CreateLambda(
    [](heavy::Context& Context, ValueRefs) {
      Value HandleError = Context.getCapture(0);
      Value MainThunk = Context.getCapture(1);
      Context.WithExceptionHandler(HandleError, MainThunk);
    }, CaptureList{HandleError, MainThunk}));

  // Run the loop.
  Context.Resume();
}

void HeavyScheme::RegisterModule(llvm::StringRef MangledName,
                                 heavy::ModuleLoadNamesFn* LoadNamesFn) {
  getContext().RegisterModule(MangledName, LoadNamesFn);
}

heavy::FullSourceLocation
HeavyScheme::getFullSourceLocation(heavy::SourceLocation Loc) {
  return getSourceManager().getFullSourceLocation(Loc);
}

heavy::Value HeavyScheme::ParseSourceFile(uintptr_t ExternalRawLoc,
                                          llvm::StringRef Name,
                                          char const* BufferStart,
                                          char const* BufferEnd,
                                          char const* BufferPos) {
  heavy::Lexer Lexer = createEmbeddedLexer(ExternalRawLoc, Name,
                                           BufferStart, BufferEnd,
                                           BufferPos);
  return ParseSourceFile(Lexer);
}

heavy::Value HeavyScheme::ParseSourceFile(heavy::Lexer Lexer) {
  heavy::Parser Parser(Lexer, getContext());
  heavy::ValueResult Result = Parser.Parse();
  if (Parser.HasError()) {
    Parser.RaiseError();
    return Undefined();
  }
  return Result.get();
}

}
