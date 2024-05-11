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

#include <heavy/Builtins.h>
#include <heavy/Lexer.h>
#include <heavy/Parser.h>
#include <heavy/HeavyScheme.h>
#include <heavy/SourceManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Process.h>
#include <string>
#include <system_error>

namespace cl = llvm::cl;

enum class ExecutionMode {
  repl,
  read,
  mlir,
};

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"));

static cl::opt<ExecutionMode> InputMode(
  "mode", cl::desc("mode of execution"),
  cl::values(
    cl::OptionEnumValue{"repl",
                        (int)ExecutionMode::repl,
                        "read eval and print loop (not interactive yet)"},
    cl::OptionEnumValue{"read",
                        (int)ExecutionMode::read,
                        "just read and print"},
    cl::OptionEnumValue{"mlir",
                        (int)ExecutionMode::mlir,
                        "output mlir code"}),
  cl::init(ExecutionMode::repl));

void ProcessTopLevelExpr(heavy::Context& Context, heavy::ValueRefs Values) {
  assert(Values.size() == 2 && "expecting 2 arguments");
  heavy::Value Val = Values[0];
  heavy::Value Env = Values[1];
  switch (InputMode.getValue()) {
  case ExecutionMode::repl:
    heavy::eval(Context, Val, Env);
    return;
  case ExecutionMode::mlir:
    heavy::compile(Context, Val, Env, heavy::Undefined());
    return;
  case ExecutionMode::read:
    Val.dump();
    Context.Cont();
    return;
  default:
    llvm_unreachable("Invalid execution mode for loop");
  }
}

int main(int argc, char const** argv) {
#if 0
  // TODO Provide interactive looping which requires support
  //      in Parser/Lexer possibly. Also look at llvm::LineEditor.
  bool IsInteractive = llvm::sys::Process::StandardInIsUserInput();
#endif
  llvm::InitLLVM LLVM_(argc, argv);
  heavy::HeavyScheme HeavyScheme;
  HeavyScheme.InitSourceFileStorage();
  cl::ParseCommandLineOptions(argc, argv);
  llvm::StringRef Filename = InputFilename;
  llvm::ErrorOr<heavy::SourceFile>
    FileResult = HeavyScheme.getSourceManager().Open(Filename);
  if (std::error_code ec = FileResult.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }
  heavy::SourceFile File = FileResult.get();

  // Top level Scheme parse/eval stuff

  bool HasErrors = false;
  auto OnError = [&HasErrors](llvm::StringRef Err,
                              heavy::FullSourceLocation const& SL) {
    HasErrors = true;
    if (SL.isValid()) {
      heavy::SourceLineContext LineContext = SL.getLineContext();
      llvm::errs() << LineContext.FileName
                   << ':' << LineContext.LineNumber
                   << ':' << LineContext.Column << ": "
                   << "error: " << Err << '\n'
                   << LineContext.LineRange << '\n';
      // Display the caret pointing to the point of interest.
      for (unsigned i = 1; i < LineContext.Column; i++) {
        llvm::errs() << ' ';
      }
      llvm::errs() << "^\n";
    } else {
      llvm::errs() << "error: " << Err << "\n\n";
    }
  };

  heavy::Lexer Lexer(File);
  HeavyScheme.ProcessTopLevelCommands(Lexer, ProcessTopLevelExpr, OnError);

  if (InputMode.getValue() == ExecutionMode::mlir) {
    HeavyScheme.getContext().verifyModule();
    HeavyScheme.getContext().dumpModuleOp();
  }
  if (HasErrors) std::exit(1);
}
