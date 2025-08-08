; RUN: heavy-scheme %s 2>&1 | FileCheck %s
(import (heavy base)
        (heavy mlir)
        (nbdl comp))

(load-dialect "func")
(load-dialect "heavy")
(load-dialect "nbdl")

(define !nbdl.opaque (type "!nbdl.opaque"))
(define !nbdl.store (type "!nbdl.store"))
(define !nbdl.tag (type "!nbdl.tag"))
(define !nbdl.symbol (type "!nbdl.symbol"))
(define !nbdl.empty (type "!nbdl.empty"))
(define i32 (type "i32"))

(define (build-member-name name)
  (result
    (create-op "nbdl.member_name"
               (attributes
                 `("name", (string-attr name)))
               (result-types !nbdl.symbol))))

(%build-context
  'my_context
  1 ; num_params
  (lambda (BazArg)
    (define parent
      (result
        (create-op "nbdl.empty"
                   (result-types !nbdl.empty))))
    (define (build-member key typename . init-args)
      (define store
        (result
          (create-op "nbdl.store"
                    (attributes `("name", (flat-symbolref-attr typename)))
                    (operands init-args)
                    (result-types !nbdl.store)
                    )))
      (set! parent
        (result
          (create-op "nbdl.store_compose"
                     (operands key store parent)
                     (result-types !nbdl.store)))))
    (define foo-input
      (result
        (create-op "nbdl.literal"
                   (attributes
                     `("value", (attr "42" i32)))
                   (result-types !nbdl.opaque))))

    (build-member (build-member-name 'foo) '::moo::foo_t foo-input)
    (build-member (build-member-name 'bar) '::moo::bar_t)
    (build-member (build-member-name 'baz) '::moo::baz_t BazArg)
    (create-op "nbdl.cont"
               (operands parent))
  ))
(dump current-nbdl-module)
(verify current-nbdl-module)

; FIXME Should the symbol name be fully qualified?
; CHECK: #op{module @nbdl_gen_module {
; CHECK: "nbdl.context"() <{sym_name = "my_context"
(define my_context
  (module-lookup current-nbdl-module "my_context"))

; CHECK: ::moo::foo_t foo;
; CHECK-NEXT: ::moo::bar_t bar;
; CHECK-NEXT: ::moo::baz_t baz;
; CHECK: my_context(auto&& arg_0)
; CHECK-NEXT: : foo(42), bar(), baz(static_cast<decltype(arg_0)>(arg_0)
(translate-cpp my_context)
(newline)
