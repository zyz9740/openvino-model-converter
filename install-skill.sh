#!/bin/bash
# OpenVINO Converter Skill Installer
# Works on Windows (Git Bash), Linux, and macOS

set -e

SKILL_NAME="openvino-converter"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.claude/skills/$SKILL_NAME"

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

echo "✓ Installed to: $INSTALL_DIR"
echo ""
echo "Usage: Claude Code will auto-detect this skill"
echo "Uninstall: rm -rf $INSTALL_DIR"
