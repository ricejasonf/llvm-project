; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))
(import (heavy mlir))

(define !heavy.value (type "!heavy.value"))

; CHECK: #op{"heavy.literal"() {info = #heavy<"\22foo\22">} : () -> ()
(write (create-op "heavy.literal"
  (attributes
    `("info", (value-attr "foo")))))

(newline)

; CHECK: #op{"heavy.literal"() {info = #heavy<"5">} : () -> ()
(write (create-op "heavy.literal"
  (attributes
    `("info", (value-attr 5)))))

(newline)

; CHECK: #op{"heavy.literal"() {info = #heavy<"5000">} : () -> ()
(write (create-op "heavy.literal"
  (attributes
    `("info", (attr "#heavy<\"5000\">" !heavy.value)))))

(define the-answer
  (create-op "heavy.literal"
    (attributes
      `("info", (value-attr 41)))))

(newline)
; CHECK: non-parent:()
(write 'non-parent:)
(write (parent-op the-answer))

(newline)

; CHECK: "heavy.command"() ({
; CHECK-NEXT: %0 = "heavy.load_global"() <{name = @_HEAVYL5SheavyL4SbaseV5Swrite
; CHECK-NEXT: %1 = "heavy.literal"() {info = #heavy<"42">}
; CHECK-NEXT: "heavy.apply"(%0, %1) : (!heavy.value, !heavy.value) -> ()
; CHECK-NEXT: }) : () -> ()
(define the-number-one 1)

(define command
  (create-op "heavy.command" (regions the-number-one)))
(with-builder (lambda ()
  (at-block-begin (entry-block command))
  ((lambda ()
  (define callee
    (create-op "heavy.load_global"
      (result-types !heavy.value)
      (attributes
       `("name", (flat-symbolref-attr "_HEAVYL5SheavyL4SbaseV5Swrite")))))
  (define arg1
    (create-op "heavy.literal"
      (result-types !heavy.value)
      (attributes
       `("info", (value-attr 42)))))
  (create-op "heavy.apply"
    (operands (result callee) (result arg1)))))))
(write command)
(newline)

