// RUN: %clang_cc1 -std=c++1z -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

int a[1];
// CHECK: @a = {{.*}} global [1 x i32] zeroinitializer, align 4
template <int>
void test_transform() {
  auto [b] = a;
}
test_transform<0>;
// CHECK-LABEL: define {{.*}} _Z14test_transformILi0EEvv
// CHECK: [[ARRVAR:%.*]] = [1 x i32]
// CHECK: [[ARR:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* [[ARRVAR]], i64 0, i64 0
// CHECK-NEXT: br label %[[LOOP:.*]]
// CHECK [[LOOP]]:
// CHECK-NEXT: [[CUR:%.*]] = phi [ [[BEGIN:%.*]], {{%.*}} ] [ [[NEXT:%.*]], %[[LOOP]] ]
// CHECK-NEXT: [[DEST:%.*]] = getelementptr inbounds i32, i32* [[ARR]], i64 [[CUR]]
// CHECK-NEXT: [[SRC:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* @a, i64 0, i64 [[CUR]]
// CHECK-NEXT: [[X:%.*]] = load i32, i32*, [[SRC]]
// CHECK-NEXT: store i32 [[DEST]]
// CHECK-NEXT: [[NEXT]] = add nuw i64 [[CUR]], 1
// CHECK-NEXT: [[EQ:%.*]] = icmp eq i64 [[NEXT]], 1
// CHECK-NEXT: br i1 [[EQ]], label %[[FIN:.*]], label %[[LOOP]]
// CHECK-NEXT: [[FIN]]
// CHECK-NEXT: ret void
