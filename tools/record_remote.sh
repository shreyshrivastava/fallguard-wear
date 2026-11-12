#!/usr/bin/env bash
# Start or stop a FallGuard recording on the watch from this computer, without
# touching the watch screen.
# Usage: tools/record_remote.sh start "Walking" [serial]
#        tools/record_remote.sh stop [serial]
set -euo pipefail
ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
CMP="com.example.activitycollector/com.example.activitycollector.wear.RecordingService"
case "${1:-}" in
  start) args=(); [ -n "${3:-}" ] && args=(-s "$3")
         "$ADB" "${args[@]}" shell am start-foreground-service -n "$CMP" --es label "'${2:?label required}'" ;;
  stop)  args=(); [ -n "${2:-}" ] && args=(-s "$2")
         "$ADB" "${args[@]}" shell am stopservice -n "$CMP" ;;
  *) echo "usage: $0 start <label> [serial] | stop [serial]"; exit 1 ;;
esac
