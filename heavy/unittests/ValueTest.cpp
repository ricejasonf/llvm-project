//=== - ValueTest.cpp - Tests for heavy::Value and associated member types --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------===//

#include "heavy/HeavyScheme.h"
#include "llvm/Support/Casting.h"

#include <gtest/gtest.h>

using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

namespace heavy {

TEST(ValueTest, NullTest) {
  EXPECT_FALSE(bool(heavy::Value(nullptr)));

  {
    heavy::ValueBase* PtrNull = nullptr;
    EXPECT_FALSE(bool(heavy::Value(PtrNull)));
  }
  {
    heavy::Operation* PtrNull = nullptr;
    EXPECT_FALSE(bool(heavy::Value(PtrNull)));
  }
}

TEST(ValueTest, IntTest) {
  // Value should be "true" if it is not a nullptr
  EXPECT_TRUE(bool(heavy::Value(heavy::Int{0})))
    << "value should be true for non-nullptr";

  heavy::Value V = heavy::Int{42}; 
  ASSERT_TRUE(isa<heavy::Int>(V));
  heavy::Int I = llvm::cast<heavy::Int>(V);
  EXPECT_EQ(I, heavy::Int{42});
  EXPECT_NE(heavy::Int{12}, heavy::Int{0});
}

TEST(ValueTest, BoolTest) {
  // Value should be "true" if it is not a nullptr
  EXPECT_TRUE(bool(heavy::Value(heavy::Bool{false})))
    << "value should be true for non-nullptr";

  EXPECT_FALSE(!heavy::Value(heavy::Bool{false}))
    << "value should be true for non-nullptr";

  EXPECT_TRUE(heavy::Bool{true});
  EXPECT_FALSE(heavy::Bool{false});

  heavy::Value V = heavy::Bool{true}; 
  ASSERT_TRUE(isa<heavy::Bool>(V));
  heavy::Bool B = cast<heavy::Bool>(V);
  EXPECT_EQ(B, heavy::Bool{true});

  // isTrue is for Scheme conditional expressions
  V = nullptr;
  EXPECT_TRUE(V.isTrue());
  V = heavy::Bool{true};
  EXPECT_TRUE(V.isTrue());
  V = heavy::Int{0};
  EXPECT_TRUE(V.isTrue());
  V = heavy::Bool{false};
  EXPECT_FALSE(V.isTrue());

}

}
