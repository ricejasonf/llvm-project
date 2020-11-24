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
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <string>
#include <system_error>

namespace cl = llvm::cl;

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::Required);

clang::SourceLocation getMainFileLoc(clang::SourceManager& SM) {
  clang::FileID MainFileId = SM.getMainFileID();
  if (!MainFileId.isValid()) return {};

  return SM.getLocForStartOfFile(MainFileId);
}

int main(int argc, char const** argv) {
  llvm::InitLLVM LLVM_(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  llvm::StringRef Filename = InputFilename;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code ec = file.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }
  llvm::StringRef FileBuffer = file.get()->getBuffer();
  clang::SourceManagerForFile SMFF(Filename, FileBuffer);

  clang::SourceManager& SM = SMFF.get();
  clang::DiagnosticsEngine& Diags = SM.getDiagnostics();

  unsigned ErrorId = Diags.getCustomDiagID(
                      clang::DiagnosticsEngine::Level::Error,
                      /*FormatString=*/"%0");

  // Top level Scheme parse/eval stuff

  heavy::Context Context{};
  heavy::Lexer   SchemeLexer(getMainFileLoc(SM), FileBuffer);
  heavy::Parser  Parser(SchemeLexer, Context);

  Parser.ConsumeToken();

  // Do we need to create an environment here ?
  //Context.EnvStack = ???;

  heavy::ValueResult Result;
  bool HasError = Context.CheckError();
  while (true) {
    if (!HasError && Context.CheckError()) {
      HasError = true;
      Diags.Report(Context.getErrorLocation(), ErrorId)
        << Context.getErrorMessage();
    }
    // Keep parsing until we find the end
    // brace (represented by isUnset() here)
    Result = Parser.ParseTopLevelExpr();
    if (Result.isUnset()) break;
    if (HasError) continue;
    if (Result.isUsable()) {
      heavy::Value* Val = eval(Context, Result.get());
      // TODO just discard the value without dumping it
      if (!Context.CheckError()) Val->dump();
    }
  };
}
