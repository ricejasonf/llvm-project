//===-- heavy_main.cpp - HeavyScheme ------------ --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the heavy scheme interpreter.
//
//===----------------------------------------------------------------------===//

#include <heavy/Lexer.h>
#include <heavy/Parser.h>
#include <heavy/HeavyScheme.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <string>
#include <system_error>

namespace cl = llvm::cl;

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::Required);

int main(int argc, char const** argv) {
  llvm::InitLLVM LLVM_(argc, argv);
  heavy::SourceManager SourceMgr{};
  cl::ParseCommandLineOptions(argc, argv);
  llvm::StringRef Filename = InputFilename;
  llvm::ErrorOr<heavy::SourceFile> FileResult = SourceMgr.Open(Filename);
  if (std::error_code ec = FileResult.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }
  heavy::SourceFile File = FileResult.get();

  // Top level Scheme parse/eval stuff

  heavy::Context Context{};
  heavy::Lexer   SchemeLexer(File.StartLoc, File.Buffer);
  heavy::Parser  Parser(SchemeLexer, Context);

  Parser.ConsumeToken();

  // Make the Top Level Environment mutable
  Context.EnvStack = Context.CreatePair(Context.CreateModule(),
                                        Context.EnvStack);

  heavy::ValueResult Result;
  bool HasError = Context.CheckError();
  while (true) {
    Result = Parser.ParseTopLevelExpr();
    if (!HasError && Context.CheckError()) {
      HasError = true;
      // TODO print error source location
      // with Context.getErrorLocation()
      llvm::errs() << "\nerror: "
                   << Context.getErrorMessage()
                   << "\n\n";
    }
    // Keep parsing until we find the end
    if (Parser.isFinished()) break;
    if (HasError) continue;
    if (Result.isUsable()) {
      heavy::Value* Val = eval(Context, Result.get());
      // TODO just discard the value without dumping it
      if (!Context.CheckError()) Val->dump();
    }
  };
}
