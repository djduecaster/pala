# Portfolio media

The featured film is **PALA V1 Demo** (1:47). Publish it as the GitHub Release asset
`pala-v1-demo.mp4`; keep the binary out of source history. The README thumbnail is
an intentionally small tracked JPEG extracted from the approved film.

## Local production archive

These paths are relative to this repository and ignored by Git:

- `logs/local/media/pala-v1-demo.mp4`: approved master with public-facing filename.
- `logs/local/media/production-outputs/`: production exports, alternative cuts, editable
  render modules, manifests and build scripts, preserving original internal names.
- `logs/local/media/source-footage/`: supplied phone footage and hardware photographs.
- `logs/local/media/inventory.json`: original-to-local copy inventory.
- `logs/portfolio_review/`: additional alternatives, music, simulator captures,
  review sheets and QC records.
- `logs/local/archive/docs_cleanup_20260919/`: retired planning notes and guide backups.

These materials are retained locally, not included in a fresh clone. Original
external copies remain intact. Ignored files are not a backup; preserve the local
archive separately before deleting a working checkout or using `git clean -x`.

## Publication checklist

1. Review and commit the V1 source/documentation changes.
2. Review the prepared GitHub release draft and its `pala-v1-demo.mp4` asset.
3. Set the release to the intended V1 commit and publish it when ready.
4. Check playback/download from the README release link.

The film includes reconstructed commanded-motion charts, not measured feedback.
Preserve that distinction in captions, release notes, and future portfolio reuse.
