; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s

; CHECK: "foo"
"foo"
; CHECK: "bar"
"bar"
; CHECK: "foo\\nbar"
"foo\\nbar"
; CHECK: "escape\nnewline"
"escape\nnewline"
; CHECK: "\a\b\t\n\r\"\\\|"
"\a\b\t\n\r\"\\\|"

; All relevant hex codes should
; normalize to relevant escape codes
; or a hex code if there is no escape code.
; Normalize to uppercase escape code.
; CHECK: "\a\b\t\n\r\\"
"\x007;\x8;\x9;\xA;\xd;\x5C;"

; Normalize to uppercase escape code
; with no leading zeros.
; CHECK: "\xC;"
"\x00c;"

; Normalize printable UTF8 codepoints.
; CHECK: "☕"
"\x2615;"
