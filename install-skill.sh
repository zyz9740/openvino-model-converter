#!/bin/bash
# OpenVINO Converter Skill Installer
# Works on Windows (Git Bash), Linux, and macOS

set -e

SKILL_NAME="openvino-converter"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect platform and set correct .claude path
if [ -n "$USERPROFILE" ]; then
    # Windows - use USERPROFILE and convert to Unix path
    CLAUDE_DIR="$(cygpath -u "$USERPROFILE")/.claude" 2>/dev/null || CLAUDE_DIR="$HOME/.claude"
else
    # Linux/macOS
    CLAUDE_DIR="$HOME/.claude"
fi

INSTALL_DIR="$CLAUDE_DIR/skills/$SKILL_NAME"

echo "Installing $SKILL_NAME skill..."

# Check SKILL.md exists
if [ ! -f "$SCRIPT_DIR/SKILL.md" ]; then
    echo "Error: SKILL.md not found"
    exit 1
fi

# Create install directory
mkdir -p "$INSTALL_DIR"

# Copy files
cp "$SCRIPT_DIR/SKILL.md" "$INSTALL_DIR/"
[ -d "$SCRIPT_DIR/scripts" ] && cp -r "$SCRIPT_DIR/scripts" "$INSTALL_DIR/"
[ -d "$SCRIPT_DIR/references" ] && cp -r "$SCRIPT_DIR/references" "$INSTALL_DIR/"

# Show installed location (convert back to Windows path if on Windows)
if [ -n "$USERPROFILE" ]; then
    WIN_PATH="$(cygpath -w "$INSTALL_DIR" 2>/dev/null || echo "$INSTALL_DIR")"
    echo "✓ Installed to: $WIN_PATH"
else
    echo "✓ Installed to: $INSTALL_DIR"
fi

echo ""
echo "Usage: Claude Code will auto-detect this skill"
echo "Uninstall: rm -rf \"$INSTALL_DIR\""
