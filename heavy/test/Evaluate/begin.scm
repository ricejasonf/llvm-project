; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

; Test top level begin.
; CHECK: foo
; CHECK-NEXT: bar
(begin
  (write 'foo)
  (newline)
  (write 'bar)
  (newline))
