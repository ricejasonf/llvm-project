(import (heavy builtins))

(define-library (heavy mlir)
  (import (heavy base)
          (heavy mlir builtins))
    (begin
    (define (init-regions Op BlockArgTypesList UserFns)
      (define Is (range (length BlockArgTypesList)))
      (define (InitRegion RegionIndex BlockArgTypes UserFn)
        (with-builder
          (lambda ()
            (define Region (get-region Op RegionIndex))
            (define Block (entry-block Region))
            (define Args
              (let ()
                (define (Proc BlockArgType)
                  (add-argument Block BlockArgType Loc))
                (map Proc BlockArgTypes)))
            (at-block-begin Block)
            (apply UserFn Args))))
      (map InitRegion Is BlockArgTypesList UserFns))

    (define-syntax create-op
      (syntax-rules (: loc: attributes: operands: result-types: region:)
        ((create-op Name
          (loc: Loc)
          (operands: Operands ...)
          (attributes: (AttrName Attr) ...)
          (result-types: ResultTypes ...)
          (region: RegionName ((BlockArg : BlockArgType) ...)
                    RegionBody ...) ...)
         (let ((Op
                  (old-create-op Name
                    (loc Loc)
                    (operands Operands ...)
                    (attributes (list 'AttrName Attr) ...)
                    (result-types ResultTypes ...)
                    (regions (length '(RegionName ...)))
                    ))
                (BlockArgsTypesList (list (list BlockArgType ...) ...))
                (UserFns (list (lambda (BlockArg ...)
                                     RegionBody ...) ...)))
            (init-regions Op BlockArgsTypesList UserFns)
            Op)
        )))
    ) ; end of begin

   (export
    create-op
    ; (heavy mlir builtins)
    old-create-op
    current-builder
    get-region
    entry-block
    add-argument
    results
    result
    at-block-begin
    at-block-end
    block-op
    op-next
    parent-op
    set-insertion-point
    set-insertion-after
    type
    %function-type
    attr
    type-attr
    value-attr
    string-attr
    flat-symbolref-attr
    with-new-context
    with-builder
    load-dialect
    verify
    module-lookup
    value?
    )
   )

