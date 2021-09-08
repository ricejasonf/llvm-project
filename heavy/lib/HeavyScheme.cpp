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

void HeavyScheme::SetEnvironment(Environment& Env) {
  auto& Context = getContext();
  Context.setEnvironment(&Env);
}

bool HeavyScheme::ProcessTopLevelCommands(
                              heavy::Lexer& Lexer,
                              llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                              heavy::tok Terminator) {
  return ProcessTopLevelCommands(Lexer, base::eval, ErrorHandler, Terminator);
}

bool HeavyScheme::ProcessTopLevelCommands(
                              heavy::Lexer& Lexer,
                              llvm::function_ref<ValueFnTy> ExprHandler,
                              llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                              heavy::tok Terminator) {
  auto& Context = getContext();
  auto& SM = getSourceManager();
  auto handleError = [&] {
    heavy::FullSourceLocation FullLoc = SM.getFullSourceLocation(
        Context.getErrorLocation());
    ErrorHandler(Context.getErrorMessage(), FullLoc);
    return true;
  };

  heavy::Parser Parser(Lexer, Context);
  if (!Parser.PrimeToken(Terminator)) {
    handleError();
    return true;
  }

  Context.SetErrorHandler(Context.CreateLambda(
    [handleError](heavy::Context& C, ValueRefs Args) {
      if (!C.CheckError()) {
        Value Obj = Args[0];
        std::string Msg;
        llvm::raw_string_ostream Stream(Msg);
        write(Stream << "uncaught object: ", Obj);
        C.SetError(Msg, Obj);
        return;
      }
      handleError();
      C.Cont(Undefined());
    }, /*Captures=*/{}));

  heavy::Lambda* Lambda = Context.CreateLambda(ExprHandler, {});

  heavy::ValueResult Result;
  while (!Parser.isFinished()) {
    Result = Parser.ParseTopLevelExpr();
    if (Result.isUsable()) {
      heavy::Value ResultVal = Result.get();
      Context.Apply(Lambda, ResultVal);
      Context.Resume();
    }
  };

  return Context.CheckError();
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
