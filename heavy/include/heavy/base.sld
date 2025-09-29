(import (heavy builtins))

(define-library (heavy base)
  (import (heavy builtins))
  (import (heavy base r7rs-syntax))
  (begin
    (define (caar x) (car (car x)))
    (define (cadr x) (car (cdr x)))
    (define (cdar x) (cdr (car x)))
    (define (cddr x) (cdr (cdr x)))

    (define (member-fast obj list compare)
      (if (pair? list)
        (if (compare obj (car list))
          list
          (member-fast obj (cdr list) compare))
        #f))

    ; TODO Use builtins.case-lambda.
    (define (member obj list . optional-compare)
      (define compare
        (if (pair? optional-compare)
          (car optional-compare)
          equal?))
      (member-fast obj list compare))




    ) ; end of begin
  (export
    ; syntax
    define
    define-syntax
    if
    lambda
    quasiquote
    quote
    set!
    syntax-rules
    ir-macro-transformer
    begin
    cond-expand
    define-library
    export
    include
    include-ci
    include-library-declarations
    source-loc
    parse-source-file
    +
    -
    /
    *
    >
    <
    apply
    append
    call-with-values
    call/cc
    car
    cdr
    cons
    dump
    dynamic-wind
    eq?
    equal?
    eqv?
    error
    length
    list
    newline
    number->string
    raise
    string-append
    string-copy
    string-length
    values
    with-exception-handler
    write

    compile
    eval
    module-path
    op-eval

    ; Type predicates
    boolean?
    bytevector?
    char?
    eof-object?
    null?
    number?
    pair?
    port?
    procedure?
    string?
    symbol?
    vector?

    ; Extended types.
    mlir-operation?
    source-value?
    )
)
