(import (heavy builtins))

(define-library (heavy base)
  (import (heavy builtins)
          (heavy base r7rs-syntax)
          (heavy base list)
          (heavy base int))
  (begin
    ; No place else to put this
    (define (not x)
      (if (eq? x #f)
        #t #f))
    ) ; end of begin
  (export
    define
    define-syntax
    if
    lambda
    case-lambda
    quasiquote
    quote
    set!
    syntax-rules
    syntax-error
    begin
    cond-expand
    define-library
    export
    include
    include-ci
    include-library-declarations
    source-loc
    source-loc-valid
    dump-source-loc
    parse-source-file
    +
    -
    /
    *
    < <= > >=
    positive? zero?
    range
    apply
    append
    call-with-values
    call/cc
    car
    cdr
    cons
    source-cons
    dump
    dynamic-wind
    eq?
    equal?
    eqv?
    error
    length
    list make-list list-set! list-ref
    vector make-vector vector-length vector-set!  vector-ref
    number->string
    raise
    string-append
    string-copy
    string-ref
    string-length
    values
    with-exception-handler
    write
    newline

    not

    ; (heavy base r7rs-syntax)
    let letrec letrec*
    cond case
    and or when unless
    do

    ; (heavy base list)
    caar cadr cdar cddr
    member memq memv
    reverse map

    ; eval stuff
    compile
    eval
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

    ; Extended types
    mlir-operation?
    source-value?

    ; Plugin stuff
    load-plugin
    load-builtin
    define-binding
    )
)
