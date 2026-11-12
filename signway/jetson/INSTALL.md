# Installing the SignWay code on the Jetson

`code/` holds the 34 files the SignWay work added or changed (laptop commit ab897bb).
The paths are relative to the repo root. The weights are NOT included: the Jetson already has them.

Assume the repo is at ~/weights. If it isn't, change REPO below.

```bash
REPO=~/weights
HERE=<path to this unzipped folder>

# 1. Back up the 4 existing files that get overwritten (the other 30 are new)
mkdir -p ~/signway_backup
for f in .gitignore README.md corridor_nav/ROBOT_COMMANDS.md corridor_nav/current/v8_drive_node.py; do
  [ -f "$REPO/$f" ] && mkdir -p ~/signway_backup/$(dirname $f) && cp "$REPO/$f" ~/signway_backup/$f
done

# 2. Copy the code over the repo
cp -r "$HERE/code/." "$REPO/"

# 3. Check that nothing was corrupted in transit
cd "$REPO" && sha256sum -c "$HERE/CHECKSUMS.txt" | grep -v ': OK$' || echo "all files OK"
```

The drive node normally runs from a copy in the home dir (~/v8_drive_node.py, see
corridor_nav/ROBOT_COMMANDS.md). Before tests T8/T9, back that copy up and replace it:
`cp ~/v8_drive_node.py ~/v8_drive_node.py.bak_pre_signway && cp $REPO/corridor_nav/current/v8_drive_node.py ~/`

API key: `cp $REPO/corridor_nav/current/signway_secrets.example ~/.signway_secrets`, then paste
the key into it with nano, then `chmod 600 ~/.signway_secrets`. The key is never in this zip.

Undo: copy the 4 files back from ~/signway_backup and delete adaptive_reasoning/, conftest.py,
and corridor_nav/current/signway_*.
