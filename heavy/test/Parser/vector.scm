; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines

#()
; CHECK:#()

#(0)
; CHECK:#(0)

#(0 "foo")
; CHECK:#(0 "foo")

#(0 1 2 "foo" 4 'five)
; CHECK:#(0 1 2 "foo" 4 (quote five))

; Support bytevectors.
#u8()
; CHECK:#u8()

#u8(0 1)
; CHECK:#u8(#x00 #x01)

#u8(0 1 2 3 4 5 255 #xFF)
; CHECK:#u8(#x00 #x01 #x02 #x03 #x04 #x05 #xFF #xFF)
