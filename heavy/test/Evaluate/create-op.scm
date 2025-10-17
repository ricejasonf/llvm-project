; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
(import (heavy builtins))
(import (heavy mlir))

(load-dialect "func")
(load-dialect "heavy")

(define !heavy.context (type "!heavy.context"))
(define !heavy.value (type "!heavy.value"))

; CHECK: #op{%0 = "heavy.literal"() {info = #heavy<"\22foo\22">} : () -> !heavy.value
(write
  (create-op "heavy.literal"
    (loc: 0)
    (operands:)
    (attributes:
      ("info" (value-attr "foo")))
    (result-types: !heavy.value)))

(newline)

; CHECK: #op{%0 = "heavy.literal"() {info = #heavy<"5">} : () -> !heavy.value
(write
  (create-op "heavy.literal"
    (loc: 0)
    (operands:)
    (attributes:
      ("info" (value-attr 5)))
    (result-types: !heavy.value)
    ))

(newline)

; CHECK: #op{%0 = "heavy.literal"() {info = #heavy<"5000">} : () -> !heavy.value
(write
  (create-op "heavy.literal"
    (loc: 0)
    (operands:)
    (attributes:
      ("info" (attr "#heavy<\"5000\">" !heavy.value)))
    (result-types: !heavy.value)))

(define the-answer
  (create-op "heavy.literal"
    (loc: 0)
    (operands:)
    (attributes:
      ("info" (value-attr 41)))
    (result-types: !heavy.value)))

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
  (create-op "heavy.command"
    (loc: 0)
    (operands:)
    (attributes:)
    (result-types:)
    (region: "body" ()
      (define callee
        (create-op "heavy.load_global"
          (loc: 0)
          (operands:)
          (attributes:
            ("name" (flat-symbolref-attr "_HEAVYL5SheavyL4SbaseV5Swrite")))
          (result-types: !heavy.value)
          ))
      (define arg1
        (create-op "heavy.literal"
          (loc: 0)
          (operands:)
          (attributes:
            ("info" (value-attr 42)))
          (result-types: !heavy.value)))
      (create-op "heavy.apply"
        (loc: 0)
        (operands: (result callee) (result arg1))
        (attributes:)
        (result-types:))
      )))
(write command)
(newline)

;; FIXME Why is it not pretty printing the func.func?
; COM-CHECK: #op{func.func @my_func(%arg0: !heavy.context, %arg1: !heavy.value) {

; CHECK: #op{"func.func"() <{
; CHECK-NEXT: ^bb0(%arg0: !heavy.context, %arg1: !heavy.value)
(define my-func
  (create-op "func.func"
    (loc: 0)
    (operands:)
    (attributes:
      ("sym_name" (string-attr "my_func"))
      ("function_type"
       (type-attr (%function-type
                    #(!heavy.context !heavy.value)
                    #()))))
    (result-types:)
    (region: "body" ((ctx : !heavy.context) (arg1 : !heavy.value))
      (define callee
        (create-op "heavy.load_global"
          (loc: 0)
          (operands:)
          (attributes:
            ("name" (flat-symbolref-attr "_HEAVYL5SheavyL4SbaseV5Swrite")))
          (result-types: !heavy.value)
          ))
      (create-op "heavy.apply"
        (loc: 0)
        (operands: ctx arg1)
        (attributes:)
        (result-types:))
      )))
(write my-func)
(newline)

