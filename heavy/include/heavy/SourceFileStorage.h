//===- SourceFileStorage.h -====================================-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains functions for opening and storing source files
//  using the file system.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_SOURCE_FILE_STORAGE_H
#define LLVM_HEAVY_SOURCE_FILE_STORAGE_H

#include "heavy/Builtins.h"
#include "heavy/Source.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace heavy {

#if 0
// SourceFileStorageNode
//  - Maintain memory address integrity to allow
//    references to be stored for lookups.
struct SourceFileStorageNode {
  SourceFileStorageNode(SourceFileStorageNode const&) = delete;

  std::unique_ptr<llvm::MemoryBuffer> Buffer;

  llvm::StringRef getBuffer() const {
    return Buffer->getBuffer();
  }
};
#endif

class SourceFileStorage {
  using NodeTy = std::unique_ptr<llvm::MemoryBuffer>;
  using StorageTy = std::vector<NodeTy>;
  using LookupTy  = llvm::StringMap<heavy::SourceLocation,
                                    llvm::BumpPtrAllocator>;

  StorageTy Storage;
  LookupTy LookupCache;

  public:
  SourceFileStorage() = default;
  llvm::ErrorOr<SourceFile> Open(heavy::SourceManager& SM,
                                 heavy::SourceLocation Loc,
                                 llvm::StringRef Filename);
};

}

#endif
