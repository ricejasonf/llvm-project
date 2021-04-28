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
#include "llvm/ADT/STLExtras.h" // function_ref
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace heavy {
class Context;
class SourceManager;

// HeavyScheme - Opaque wrapper for heavy::Context and common operations
//               needed for embedding scheme
class HeavyScheme {
  std::unique_ptr<heavy::Context> ContextPtr;
  std::unique_ptr<heavy::SourceManager> SourceManagerPtr;

  public:

  HeavyScheme(std::unique_ptr<heavy::Context>);
  // The default constructor relies on init() to lazily
  // initialize the members.
  HeavyScheme();
  ~HeavyScheme();

  // init - idempotent initializer
  void init();
  bool isInitialized() { return ContextPtr; }

  heavy::Context& getContext() {
    assert(ContextPtr && SourceManagerPtr &&
        "HeavyScheme must be initialized");
    return *ContextPtr;
  }

  heavy::SourceManager& getSourceManager() {
    assert(ContextPtr && SourceManagerPtr &&
        "HeavyScheme must be initialized");
    return *SourceManagerPtr;
  }

  heavy::Lexer createEmbeddedLexer(uintptr_t ExternalRawLoc,
                                   char const* BufferStart,
                                   char const* BufferEnd,
                                   char const* BufferPos);

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

  // LoadCoreEnv  - Loads the core environment for bootstrapping a new module
  //                This is useful as a root node when creating scope nested
  //                embedded environments.
  void LoadCoreEnv();

  // CreateTopLevelModule - Creates a new module on top of the core environment
  // and loads it as the current environment
  void CreateTopLevelModule();

  using ErrorHandlerFn = void(llvm::StringRef, heavy::FullSourceLocation);

  // ProcessTopLevelCommands
  //              - Reading tokens with the provided lexer, this command parses
  //                and evaluates a top level scheme command sequence returning
  //                true if there is an error
  //                The terminator token is only checked outside of expressions
  //                (ie a heavy::tok::r_paren can terminate without effecting
  //                 the parsing of lists that are delimited by parens)
  bool ProcessTopLevelCommands(heavy::Lexer& Lexer,
                               llvm::function_ref<ErrorHandlerFn> ErrorHandler,
                               heavy::tok Terminator = heavy::tok::eof);

};

template <typename T>
struct function_helper : function_helper<decltype(&T::operator())>
{ };

tmeplate <typename ...Xs>
struct function_helper<uintptr_t(Xs...)> {
  template <typename F>
  auto operator()(F& f, Xs ...xs) {
    f(from_value<Xs>(x)...);
  }
};


class function {
public:
  function(function const&) = default;

  // F must be trivially copyable and trivially destructible
  template <typename F>
  function(F f) {
    auto CallFn = [](F& f, auto... params
  }
};

template <typename AllocateFn, typename CallFn, typename ...Params>
class function_storage {
  AllocateFn allocate;
};

}

#endif
