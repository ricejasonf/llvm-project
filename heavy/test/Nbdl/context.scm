; RUN: heavy-scheme -module-path=%heavy_module_path %s 2>&1 | FileCheck %s
(import (heavy base)
        (heavy mlir)
        (nbdl comp))

(load-dialect "func")
(load-dialect "heavy")
(load-dialect "nbdl")

(define !nbdl.store (type "!nbdl.store"))
(define !nbdl.tag (type "!nbdl.tag"))
(define !nbdl.member_name (type "!nbdl.member_name"))
(define !nbdl.unit (type "!nbdl.unit"))
(define i32 (type "i32"))

(define (build-member-name name)
  (result
    (create-op "nbdl.member_name"
      (loc: 0)
      (operands:)
      (attributes:
        ("name" (string-attr name)))
      (result-types: !nbdl.member_name))))

(define my_store
  (with-builder
    (lambda ()
      (at-block-end (entry-block current-nbdl-module))
      (create-op "nbdl.define_store"
        (loc: 0)
        (operands:)
        (attributes: ("sym_name" (string-attr "my_store")))
        (result-types:)
        (region: "body" ((BazArg : !nbdl.store))
          (define parent
            (result
              (create-op "nbdl.unit"
                (loc: 0)
                (operands:)
                (attributes:)
                (result-types: !nbdl.unit))))
          (define (build-member key typename . init-args)
            (define store
              (result
                (create-op "nbdl.store"
                  (loc: 0)
                  (operands: init-args)
                  (attributes: ("name" (flat-symbolref-attr typename)))
                  (result-types: !nbdl.store)
                  )))
            (set! parent
              (result
                (create-op "nbdl.store_compose"
                  (loc: 0)
                  (operands: key store parent)
                  (attributes:)
                  (result-types: !nbdl.store)))))
          (define foo-input
            (result
              (create-op "nbdl.literal"
                (loc: 0)
                (operands:)
                (attributes:
                  ("value" (attr "42" i32)))
                (result-types: !nbdl.store))))
          ;; While were here test out "Any" equality.
          (unless (eq? parent parent)
            (error "expecting equal anys"))
          (when (eq? parent foo-input)
            (error "expecting unequal anys"))
          (build-member (build-member-name 'foo) '::moo::foo_t foo-input)
          (build-member (build-member-name 'bar) '::moo::bar_t)
          (build-member (build-member-name 'baz) '::moo::baz_t BazArg)
          (build-member (build-member-name 'baz2) '::moo::baz_t BazArg)
          (create-op "nbdl.cont"
            (loc: 0)
            (operands: parent)
            (attributes:)
            (result-types:))
          )))))

(define my_context
  (with-builder
    (lambda ()
      (at-block-end (entry-block current-nbdl-module))
      (create-op "nbdl.context"
        (loc: (source-loc 'my_context))
        (operands:)
        (attributes: ("sym_name" (string-attr "my_context"))
                     ("implName" (flat-symbolref-attr "my_store")))
        (result-types:)))))


(define my_store_lu
  (module-lookup current-nbdl-module "my_store"))
(define my_context_lu
  (module-lookup current-nbdl-module "my_context"))

(unless (eq? my_store my_store_lu)
  (error "expecting module lookup success" my_store_lu))
(unless (eq? my_context my_context_lu)
  (error "expecting module lookup success" my_context_lu))


; FIXME Should the symbol name be fully qualified?
; CHECK: #op{module @nbdl_gen_module {
; CHECK: "nbdl.define_store"() <{sym_name = "my_store"}>
; CHECK: "nbdl.context"() <{implName = @my_store, sym_name = "my_context"}>
(dump current-nbdl-module)
(unless (verify current-nbdl-module)
  (error "mlir verification failed"))

; CHECK: class my_store {
; CHECK: ::moo::foo_t foo;
; CHECK-NEXT: ::moo::bar_t bar;
; CHECK-NEXT: ::moo::baz_t baz;
; CHECK: my_store(auto&& arg_0)
; CHECK-NEXT: : foo(42), bar(), baz(arg_0), baz2(static_cast<decltype(arg_0)>(arg_0)) 
(translate-cpp my_store)
(translate-cpp my_context)

(newline)
