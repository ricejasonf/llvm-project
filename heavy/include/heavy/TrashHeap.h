//===---- TrashHeap.h - Garbage Collection Stuff -------------*- C++ ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  TrashHeap - Garbage collection related structures and algorithms
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_TRASH_HEAP_H
#define LLVM_HEAVY_TRASH_HEAP_H

namespace heavy {
template <typename Derived>
class TrashHeap {
  // The general idea:
  // CurrentMaxHint will double if we do not reclaim
  // at least half of the memory after scavenging.
  size_t CurrentMaxHint = 1024 * 1024; // 1 MiB
  llvm::BumpPtrAllocator Heap;
  static constexpr size_t Alignment = alignof(heavy::ValueBase*);

  Derived& getDerived() {
    return *static_cast<Derived*>(this);
  }

  protected:

  void* Allocate(size_t Size) {
    void* Ptr = Heap.Allocate(Size, Alignment);
    if (Heap.getBytesAllocated() >= CurrentMaxHint) {
      // Stop and copy.
      llvm::BumpPtrAllocator OldHeap = std::move(Heap);
      Heap = llvm::BumpAllocator();

      getDerived().VisitRoots([](heavy::Value& Val) {
        Val = Copier.Visit(Val);
      });
    }
  }
};
}

#endif // LLVM_HEAVY_TRASH_HEAP_H
