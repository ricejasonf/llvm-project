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
#include "heavy/Parser.h"
#include "heavy/Value.h"


namespace heavy {

HeavyScheme::HeavyScheme(std::unique_ptr<heavy::Context> C)
  : ContextPtr(std::move(C)),
    SourceManagerPtr(std::make_unique<heavy::SourceManager>())
{ }

HeavyScheme::HeavyScheme()
  : ContextPtr(std::make_unique<heavy::Context>()),
    SourceManagerPtr(std::make_unique<heavy::SourceManager>())
{ }

HeavyScheme::~HeavyScheme() = default;

heavy::Lexer HeavyScheme::createEmbeddedLexer(uintptr_t ExternalRawLoc,
                                              char const* BufferStart,
                                              char const* BufferEnd,
                                              char const* BufferPos) {
  heavy::SourceManager& SM = getSourceManager();
  size_t BufferLen = BufferEnd - BufferStart;
  llvm::StringRef FileBuffer(BufferStart, BufferLen);
  heavy::SourceFile File = SM.getOrCreateExternal(ExternalRawLoc, FileBuffer,
                                                  /*Name=*/"");
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
                              Environment& Env, 
                              llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                              heavy::tok Terminator) {
  return ProcessTopLevelCommands(Lexer, Env, base::eval, ErrorHandler,
                                 Terminator);
}

void HeavyScheme::ProcessTopLevelCommands(
                              heavy::Lexer& Lexer,
                              Environment& Env, 
                              llvm::function_ref<ValueFnTy> ExprHandler,
                              llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                              heavy::tok Terminator) {
  auto& Context = getContext();
  auto& SM = getSourceManager();
  auto HandleErrorFn = [&](heavy::Context& Context, ValueRefs) {
    assert(Context.CheckError() && "expecting hard error");
    heavy::FullSourceLocation FullLoc = SM.getFullSourceLocation(
        Context.getErrorLocation());
    ErrorHandler(Context.getErrorMessage(), FullLoc);
    Context.ClearStack();
  };

  heavy::Parser Parser(Lexer, Context);
  if (!Parser.PrimeToken(Terminator)) {
    HandleErrorFn(Context, {});
    return;
  }

  heavy::ExternLambda<0, sizeof(ExprHandler)> HandleExpr;
  heavy::ExternLambda<0, sizeof(ErrorHandler)> HandleError;
  heavy::ExternLambda<0, (sizeof(void*) * 4)> MainThunk;

  auto MainThunkFn = [&Parser, &MainThunk, &HandleExpr, &Env]
                          (heavy::Context& C, ValueRefs) {
    assert(!C.CheckError() && "Error should have escaped.");
    if (Parser.isFinished()) {
      C.Cont();
      return;
    }

    // Recurse in tail position.
    C.PushCont(Value(MainThunk));

    heavy::ValueResult ParseResult = Parser.ParseTopLevelExpr();
    if (ParseResult.isUsable()) {
      Value ParseResultVal = ParseResult.get();
      std::array<Value, 2> EvalArgs{ParseResultVal, &Env};
      C.Apply(Value(HandleExpr), EvalArgs);
    } else {
      C.Cont();
    }
  };

  // Store the handlers and thunk.
  HandleExpr  = ExprHandler;
  HandleError = HandleErrorFn;
  MainThunk   = MainThunkFn;

  // Run the loop.
  Context.WithExceptionHandlers(HandleError, MainThunk);
  Context.Resume();
}

void HeavyScheme::RegisterModule(llvm::StringRef MangledName,
                                 heavy::ModuleLoadNamesFn* LoadNamesFn) {
  getContext().RegisterModule(MangledName, LoadNamesFn);
}

heavy::Undefined setError(heavy::Context& C, llvm::StringRef Msg) {
  C.SetError(Msg);
  return {};
}

}
