import lldb
import os
import sys
from itertools import zip_longest

# During run-time, detect the locations where a
# continuation is set multiple times before continuing.
# Normally, this would trigger and assert in
# ContinuationStack<T>::ApplyHelper.
#
# This scripts output is two stack trace lists of
# source locations side by side to see where the two
# continuation calls were made.

backtraces = []

def save_frames(frames):
    global backtraces
    result = []
    for frame in frames:
        entry = frame.line_entry
        filename = entry.GetFileSpec().GetFilename()
        line = entry.line
        column = entry.column
        result.append(f'  {filename}:{line}:{column}')
    backtraces.append(result)

def print_frames(title1, frames1, title2, frames2):
    print(f'{title1:35}{title2:35}')
    for frame_row in reversed(list(zip_longest(reversed(frames1),
                                               reversed(frames2), fillvalue=" "))):
        print(f'{frame_row[0]:35}{frame_row[1]:35}')

def on_ApplyHelper(frame):
    global backtraces
    # Save the backtrace to record locations of previous calls.
    thread = frame.GetThread()
    did_call_twice = frame.EvaluateExpression("DidCallContinuation").unsigned
    save_frames(thread.frames)
    if did_call_twice:
        print_frames("Previous Call:", backtraces[-1], "Last Call:", backtraces[-2])
    thread.process.Continue()

def main():
    build_dir = os.getcwd()
    debugger = lldb.SBDebugger.Create()
    debugger.SetAsync(False)

    target = debugger.CreateTarget(build_dir + "/bin/heavy-scheme")
    if not target:
        print('failed to create target')
        exit(1)
    break_ApplyHelper = target.BreakpointCreateByRegex("ContinuationStack.*ApplyHelper")
    break_ApplyHelper.SetEnabled(True)
    assert(break_ApplyHelper.GetNumLocations() > 0)

    break_ApplyHelper.SetScriptCallbackBody("on_ApplyHelper(frame)")

    process = target.LaunchSimple(sys.argv[1:], None, os.getcwd())
    process.Continue()
    thread = process.GetThreadAtIndex(0)
    frame = thread.GetFrameAtIndex(0)

if __name__ == "__main__":
    main()




