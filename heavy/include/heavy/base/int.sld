(import (heavy builtins))

(define-library (heavy base int)
  (import (heavy builtins)
          (heavy base r7rs-syntax))
  (begin
    (define (< x1 x2 . xN)
      (if (positive? (- x2 x1))
        (if (pair? xN)
          (apply < x2 xN)
          #t)
        #f))

    (define (<= x1 x2 . xN)
      (let ((Difference (- x2 x1)))
        (if (or (positive? Difference) (zero? Difference))
          (if (pair? xN)
            (apply <= x2 xN)
            #t)
          #f)))

    (define (> x1 x2 . xN)
      (if (positive? (- x1 x2))
        (if (pair? xN)
          (apply < x2 xN)
          #t)
        #f))

    (define (>= x1 x2 . xN)
      (let ((Difference (- x1 x2)))
        (if (or (positive? Difference) (zero? Difference))
          (if (pair? xN)
            (apply >= x2 xN)
            #t)
          #f)))

    ) ; end begin
  (export
    < <= > >=
    ))
