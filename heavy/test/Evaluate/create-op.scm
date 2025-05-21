; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))
(import (heavy mlir))

(define !heavy.value (type "!heavy.value"))

; CHECK: #op{"heavy.literal"() {info = #heavy<"\22foo\22">} : () -> ()
(write (create-op "heavy.literal"
  (attributes
    `("info", (attr "foo")))))

(newline)

; CHECK: #op{"heavy.literal"() {info = #heavy<"5">} : () -> ()
(write (create-op "heavy.literal"
  (attributes
    `("info", (attr 5)))))

(newline)

; CHECK: #op{"heavy.literal"() {info = #heavy<"5000">} : () -> ()
(write (create-op "heavy.literal"
  (attributes
    `("info", (attr !heavy.value "#heavy<\"5000\">")))))

(define the-answer
  (create-op "heavy.literal"
    (attributes
      `("info", (attr 41)))))

(newline)
; CHECK: non-parent:()
(write 'non-parent:)
(write (parent-op the-answer))

(newline)

; CHECK "heavy.command"()({
; CHECK-NEXT %1 = "heavy.load_global"() <{name = @_HEAVYL5SheavyL4SbaseV5Swrite
; CHECK-NEXT %2 = "heavy.literal"() <{input = #heavy<"42">
; CHECK-NEXT "heavy.apply"(%1, %2) : (!heavy.value, !heavy.value) -> ()
; CHECK-NEXT }) : () -> ()
(define command
  (create-op "heavy.command"))

(set-insertion-point command)

(define callee
  (create-op "heavy.load_global"
    (result-types !heavy.value)
    (attributes
     `("name", (attr "_HEAVYL5SheavyL4SbaseV5Swrite")))))

(define arg1
  (create-op "heavy.literal"
    (result-types !heavy.value)
    (attributes
     `("info", (attr 42)))))

(create-op "heavy.apply"
  (operands (result callee) (result arg1)))

