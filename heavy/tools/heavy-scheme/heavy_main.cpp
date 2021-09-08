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
  assert(Values.size() == 1 && "Expecting single parse result");
  heavy::Value Val = Values[0];
  switch (InputMode.getValue()) {
  case ExecutionMode::repl:
    Val = heavy::eval(Context, Val);
    return;
  case ExecutionMode::read:
    if (!Context.CheckError()) Val.dump();
    break;
  case ExecutionMode::mlir:
    Context.PushTopLevel(Val);
    break;
  default:
    llvm_unreachable("Invalid execution mode for loop");
  }
  Context.Cont();
}

int main(int argc, char const** argv) {
#if 0
  // TODO Provide interactive looping which requires support
  //      in Parser/Lexer possibly. Also look at llvm::LineEditor.
  bool IsInteractive = llvm::sys::Process::StandardInIsUserInput();
#endif
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

  auto OnError = [](llvm::StringRef Err, heavy::FullSourceLocation) {
    // TODO display error location
    llvm::errs() << "\nerror: "
                 << Err
                 << "\n\n";
  };

  auto Env = std::make_unique<heavy::Environment>();
  heavy::HeavyScheme HeavyScheme;
  heavy::Lexer Lexer(File);
  HeavyScheme.SetEnvironment(*Env);
  bool HasErrors = HeavyScheme.ProcessTopLevelCommands(Lexer, ProcessTopLevelExpr, OnError);

  if (InputMode.getValue() == ExecutionMode::mlir) {
    HeavyScheme.getContext().dumpModuleOp();
    llvm::errs() << "\n";
  }
  if (HasErrors) std::exit(1);
}
