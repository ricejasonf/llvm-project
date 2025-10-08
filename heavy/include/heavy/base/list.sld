(import (heavy builtins))

(define-library (heavy base list)
  (import (heavy builtins)
          (heavy base r7rs-syntax)
          (only (heavy base int) < <=))
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
    (define member
      (case-lambda
        ((member obj list compare)
            (member-fast obj list compare))
        ((member obj list)
            member obj list equal?)))
    (define (memq obj list)
      (member-fast obj list eq?))
    (define (memv obj list)
      (member-fast obj list eqv?))

    (define (reverse InputList)
      (let Loop ((List InputList)
                 (NewList '()))
        (if (pair? List)
          (Loop (cdr List)
            (cons (car List) NewList))
          NewList)))

    (define (map Proc . InputLists)
      (define (MapFast FastProc List)
        (if (pair? List)
          (cons (FastProc (car List))
                (MapFast FastProc (cdr List)))
          '()))
      (define MaxLen
        (let Loop ((Lists InputLists)
                   (MinLength -1))
          (dump Lists)
          (if (pair? Lists)
            (let ((Len (length (car Lists))))
              (Loop
                (cdr Lists)
                (if (<= 0 Len MinLength)
                  Len
                  MinLength)))
            MinLength)))
      (dump 'wtf)
      (when (< MaxLen 0)
        (error "expecting at least one finite list" InputLists))
      (let Loop ((I 0)
                 (Lists InputLists)
                 (Result '()))
        (if (< I MaxLen)
          (let ((Args (MapFast car Lists))
                (NextLists (MapFast cdr Lists)))
            (Loop (+ I 1) NextLists (cons (apply Proc Args))))
          Result)))



    ) ; end of begin
  (export
    caar cadr cdar cddr
    member memq memv
    reverse map
    )
)
