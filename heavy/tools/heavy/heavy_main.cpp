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

#include <heavy/HeavyScheme.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>

namespace cl = llvm::cl;

SourceLocation getMainFileLoc(SourceManager& SM) {
  FileId MainFileId = SM.getMainFileID();
  if (!MainFileId.isValid()) return {};

  return SM.getLocForStartOfFile(MainFileId);
}

int main(int argc, char const** argv) {
  llvm::InitLLVM LLVM_(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  cl::opt<string> InputFilename(cl::Positional,
                                cl::desc("<input file>"),
                                cl::init("-"));
  StringRef Filename = InputFileName;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
  if (file.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }
  StringRef FileBuffer = fileOrErr.get()->getBuffer();
  clang::SourceManagerForFile SMFF(Filename, FileBuffer);

  clang::DiagnosticEngine& Diag = SMFF.Diag;

  // Top level Scheme parse/eval stuff

  auto SchemeLexer = HeavySchemeLexer(getMainFileLoc(), FileBuffer);

  heavy::Context Context = {};

  auto SchemeLexer = HeavySchemeLexer();
  // init HeavyScheme Lexer to the buffer of the main file

  ParserHeavyScheme P(SchemeLexer, Context, *this);

  P.ConsumeToken();

  // Do we need to create an environment here ?
  //Context.EnvStack = ???;

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
}
