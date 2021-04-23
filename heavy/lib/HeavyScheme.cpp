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

#include "heavy/Context.h"
#include "heavy/HeavyScheme.h"
#include "heavy/Lexer.h"
#include "heavy/Parser.h"


namespace heavy {

HeavyScheme::HeavyScheme(std::unique_ptr<heavy::Context> C)
  : ContextPtr(std::move(C)),
    SourceManagerPtr(std::make_unique<heavy::SourceManager>())
{ }

HeavyScheme::HeavyScheme() = default;
HeavyScheme::~HeavyScheme() = default;

void HeavyScheme::init() {
  if (!ContextPtr) {
    ContextPtr = std::make_unique<heavy::Context>();
    SourceManagerPtr = std::make_unique<heavy::SourceManager>();
  }
}

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
  auto itr = Context.EmbeddedEnvs.find(Handle);
  if (itr != Context.EmbeddedEnvs.end()) {
    Context.EnvStack = itr->second;
    return;
  }

  LoadParent(*this, Handle);
  Value Env = Context.EnvStack;
  Env = Context.CreatePair(Context.CreateModule(), Env);
  Context.EmbeddedEnvs[Handle] = Env;
  Context.EnvStack = Env;
  return;
}

void HeavyScheme::LoadCoreEnv() {
  auto& Context = getContext();
  Context.EnvStack = Context.SystemEnvironment;
}

void HeavyScheme::CreateTopLevelModule() {
  auto& Context = getContext();
  LoadCoreEnv();
  Context.PushMutableModule();
}

bool HeavyScheme::ProcessTopLevelCommands(
                              heavy::Lexer& Lexer,
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

  heavy::ValueResult Result;
  bool HasError = Context.CheckError();
  while (true) {
    Result = Parser.ParseTopLevelExpr();
    if (!HasError && Context.CheckError()) {
      HasError = true;
      handleError();
    }
    // Keep parsing until we find the end
    if (Parser.isFinished()) break;
    if (HasError) continue;
    if (Result.isUsable()) {
      heavy::Value ResultVal = Result.get();
      Context.HandleParseResult(Context, ResultVal);
    }
  };

  return HasError;
}

}
