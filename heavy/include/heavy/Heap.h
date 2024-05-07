//===--- Heap.h - Classes for representing declarations ----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares heavy::Heap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_HEAP_H
#define LLVM_HEAVY_HEAP_H

#include "heavy/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>

namespace heavy {
template <typename Derived>
class Heap;
}

template <typename Derived>
void* operator new(size_t Size, heavy::Heap<Derived>& Heap);
template <typename Derived>
void operator delete(void*, heavy::Heap<Derived>&) { }


namespace heavy {
using llvm::ArrayRef;
using llvm::StringRef;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

// Implement CopyCollector in Heap.cpp.
class CopyCollector;

// Derived implements CollectGarbage() which calls
// VisitRootGC(Value) for each root node.
template <typename Derived>
class Heap : public llvm::AllocatorBase<Heap<Derived>> {
  friend Derived;
  using AllocatorTy = llvm::BumpPtrAllocator;

  Derived& getDerived() {
    return *static_cast<Derived*>(this);
  }

  AllocatorTy TrashHeap;

  // MaxHint - The threshold used to determine if a garbage
  //           collection run is needed. This value is not a
  //           hard limit and the limit is increased when a
  //           collection run yields a low return. The limit
  //           may also increase with the allocation of a large
  //           object.
  size_t MaxHint;

public:
  Heap(size_t MaxStart)
    : MaxHint(MaxStart)
  { }

  // getBytesAllocated
  //    - Provide the number of bytes allocated on the
  //      heap. This does not include intermediate data
  //      stored on an old heap that is being scavenged.
  size_t getBytesAllocated() const {
    return TrashHeap.getBytesAllocated();
  }

  // Allocate a potentially large object (like a String).
  void* BigAllocate(size_t Size, size_t Alignment) {
    size_t WorstCase = Size + Alignment;
    size_t BytesUsed = getBytesAllocated();
    if (WorstCase >= Derived::MiB && double(WorstCase) / double(BytesUsed) > 0.25) {
      // Just increase the MaxHint since GC will not
      // likely do much in this scenario.
      MaxHint += WorstCase;
    }

    return Allocate(Size, Alignment);
  }

  void* Allocate(size_t Size, size_t Alignment) {
    size_t WorstCase = Size + Alignment;
    size_t BytesUsed = getBytesAllocated();

    if (BytesUsed + WorstCase > MaxHint) {
      getDerived().CollectGarbage();
    }

    // Allocate on the possibly new heap.
    return TrashHeap.Allocate(Size, Alignment);
  }

  void Deallocate(void const*, size_t, size_t) {
    // Do nothing.
  }
};

class IdTable {
  using AllocatorTy = llvm::BumpPtrAllocator;

  // Provide a special heap for identifiers
  // so their pointers are stable to be used
  // as keys in maps like Module, Context, etc.
  AllocatorTy IdHeap;
  llvm::StringMap<String*> IdTableMap = {};
  std::string Buffer = {};

public:
  String* CreateIdTableEntry(llvm::StringRef S);
  String* CreateIdTableEntry(llvm::StringRef Prefix, llvm::StringRef S);
};

}  // namespace heavy

template <typename Derived>
void* operator new(size_t Size, heavy::Heap<Derived>& Heap) {
  return Heap.Allocate(Size,
    std::min((size_t)llvm::NextPowerOf2(Size), alignof(std::max_align_t)));
}

#endif
