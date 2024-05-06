//===- HeavyScheme.h - Classes for representing declarations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains opaque user interfaces for HeavyScheme
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_HEAVYSCHEME_H
#define LLVM_HEAVY_HEAVYSCHEME_H

#include "heavy/Lexer.h"
#include "heavy/Source.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h" // function_ref
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace heavy {
class Context;
class SourceFileStorage;
class SourceManager;
class Value;
class Undefined;
class Environment;
class FullSourceLocation;
using ModuleLoadNamesFn = void(heavy::Context&);
using ValueRefs   = llvm::MutableArrayRef<heavy::Value>;
using ValueFnTy   = void (Context&, ValueRefs);

// HeavyScheme - Opaque wrapper for heavy::Context and common operations
//               needed for embedding scheme
class HeavyScheme {
  friend class SourceFileStorage;

  std::unique_ptr<heavy::Context> ContextPtr;
  std::unique_ptr<heavy::Environment> EnvPtr;
  std::unique_ptr<heavy::SourceManager> SourceManagerPtr;
  std::unique_ptr<heavy::SourceFileStorage,
                  void(*)(SourceFileStorage*)> SourceFileStoragePtr;

  heavy::SourceManager& getSourceManager() {
    assert(SourceManagerPtr && "HeavyScheme must be initialized");
    return *SourceManagerPtr;
  }

  heavy::SourceFileStorage& getSourceFileStorage() {
    assert(SourceFileStoragePtr &&
        "InitSourceFileStorage must be called for "
        "optional file system support");
    return *SourceFileStoragePtr;
  }

  public:

  HeavyScheme(std::unique_ptr<heavy::Context>);
  HeavyScheme();
  ~HeavyScheme();

  heavy::Context& getContext() {
    assert(ContextPtr && "HeavyScheme must be initialized");
    return *ContextPtr;
  }

  heavy::FullSourceLocation getFullSourceLocation(heavy::SourceLocation Loc);

  heavy::Lexer createEmbeddedLexer(uintptr_t ExternalRawLoc,
                                   llvm::StringRef Name,
                                   char const* BufferStart,
                                   char const* BufferEnd,
                                   char const* BufferPos);
  heavy::Value ParseSourceFile(heavy::Lexer Lexer);
  heavy::Value ParseSourceFile(heavy::SourceLocation Loc,
                               llvm::StringRef Filename);
  heavy::Value ParseSourceFile(uintptr_t ExternalRawLoc,
                               llvm::StringRef Name,
                               char const* BufferStart,
                               char const* BufferEnd,
                               char const* BufferPos);
  void InitSourceFileStorage();

  // LoadEmbeddedEnv
  //              - Associates an opaque pointer with a scheme environment
  //                and loads it as the current environment in the Context.
  //                Any environment previously loaded with this function is
  //                cached. If a new environment must be created,
  //                LoadParent is called, and then the new environment is
  //                created to shadow whatever environment is loaded in
  //                Context at that point.
  void LoadEmbeddedEnv(void* Handle,
          llvm::function_ref<void(HeavyScheme&, void*)> LoadParent);

  using ErrorHandlerFn = void(llvm::StringRef,
                              heavy::FullSourceLocation const&);

  // ProcessTopLevelCommands
  //              - Reading tokens with the provided lexer, this command parses
  //                and evaluates a top level scheme command sequence returning
  //                true if there is an error
  //                The terminator token is only checked outside of expressions
  //                (ie a heavy::tok::r_paren can terminate without effecting
  //                 the parsing of lists that are delimited by parens)
  //              - ExprHandler defaults to `base::eval`
  void ProcessTopLevelCommands(heavy::Lexer& Lexer,
                               llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                               heavy::tok Terminator);
  void ProcessTopLevelCommands(heavy::Lexer& Lexer,
                               llvm::function_ref<ValueFnTy> ExprHandler,
                               llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                               heavy::tok Terminator = heavy::tok::eof);
  // ProcessTopLevelCommands
  //              - Filename overload requires calling InitSourceFileStorage.
  void ProcessTopLevelCommands(llvm::StringRef Filename,
                               llvm::function_ref<ValueFnTy> ExprHandler,
                               llvm::function_ref<ErrorHandlerFn> ErrorHandler);

  // Registers a module import function
  void RegisterModule(llvm::StringRef MangledName,
                      heavy::ModuleLoadNamesFn*);
};

}

#endif
