from __future__ import annotations

import argparse
import ipaddress
import json
import os
import tempfile
from pathlib import Path

import uvicorn

from .app import create_app
from .config import load_config
from .store import Store, StoreError
from .reconstruct import build_session, build_house_session, ReconstructionError


def main():
    parser = argparse.ArgumentParser(description="Local Quest Splat capture bridge")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--export-session", help="Export a local capture manifest for a session UUID and exit")
    parser.add_argument("--export-output", type=Path, help="Manifest output path, required with --export-session")
    parser.add_argument("--build-session", help="Run installed COLMAP/Nerfstudio tools for a session UUID")
    parser.add_argument("--reviewed-native-orientation", action="store_true",
                        help="Confirm direct visual review of the Quest's native JPEG orientation")
    parser.add_argument("--quality", choices=("quick", "final", "house"), default="final",
                        help="Quick preview, detailed splat, or combined depth-assisted textured mesh")
    parser.add_argument("--fresh-baseline", action="store_true",
                        help="Start a new quick baseline from this session without resuming an older model")
    parser.add_argument("--candidate", action="store_true",
                        help="Keep this quality trial out of viewers until it is reviewed")
    parser.add_argument("--frozen-source-job-id",
                        help="Resume exactly this ready quick job's original photos for a final review candidate")
    parser.add_argument("--target-training-step", type=int,
                        help="Bounded absolute training step for a frozen final review candidate")
    parser.add_argument("--incremental-new-captures", action="store_true",
                        help="Include newly received same-session photos in a private 1000-15000-step incremental review")
    parser.add_argument("--capture-snapshot", type=Path,
                        help="Verified queued house input inventory, not arbitrary photo paths")
    args = parser.parse_args()
    if args.reviewed_native_orientation and not args.build_session:
        parser.error("--reviewed-native-orientation requires --build-session")
    if args.build_session and (args.export_session or args.export_output):
        parser.error("--build-session cannot be combined with export options")
    if args.fresh_baseline and (not args.build_session or args.quality != "quick"):
        parser.error("--fresh-baseline requires --build-session and --quality quick")
    if args.candidate and not args.build_session:
        parser.error("--candidate requires --build-session")
    if ((args.frozen_source_job_id is None) != (args.target_training_step is None) or
            (args.frozen_source_job_id is not None and
             (not args.build_session or not args.candidate or args.quality != "final" or
              args.fresh_baseline or args.capture_snapshot))):
        parser.error("Frozen continuation requires --build-session --quality final --candidate, a source job, and a target step")
    if args.incremental_new_captures and args.frozen_source_job_id is None:
        parser.error("--incremental-new-captures requires a frozen source job and target step")
    if args.quality == "house" and (not args.build_session or args.candidate or args.fresh_baseline):
        parser.error("House workflow requires --build-session and cannot use candidate/fresh-baseline flags")
    if args.capture_snapshot and args.quality != "house":
        parser.error("--capture-snapshot requires --quality house")
    if args.build_session:
        import uuid
        try:
            session_id = str(uuid.UUID(args.build_session))
        except ValueError:
            parser.error("--build-session must be a UUID")
        config = load_config(args.data_dir)
        try:
            if args.quality == "house":
                capture_ids, revision = None, None
                if args.capture_snapshot:
                    path = args.capture_snapshot.resolve()
                    expected_parent = config.data_dir / "build-queue-inputs"
                    if (args.capture_snapshot.is_symlink() or expected_parent.is_symlink() or
                            path.parent != expected_parent.resolve() or path.suffix != ".json"):
                        raise ReconstructionError("House inventory is outside the validated queue")
                    import uuid
                    uuid.UUID(path.stem)
                    inventory = json.loads(path.read_text(encoding="utf-8"))
                    session = Store(config.data_dir).session(session_id)
                    if (inventory.get("schema_version") != 1 or inventory.get("session_id") != session_id or
                            inventory.get("project_id") != session["project_id"] or
                            not isinstance(inventory.get("capture_ids"), list) or
                            not isinstance(inventory.get("input_revision"), str)):
                        raise ReconstructionError("House inventory belongs to another session")
                    capture_ids, revision = inventory["capture_ids"], inventory["input_revision"]
                delivery = build_house_session(Store(config.data_dir), session_id,
                    native_orientation_reviewed=args.reviewed_native_orientation,
                    capture_ids=capture_ids, expected_input_revision=revision)
                print(f"Depth-assisted textured mesh: {delivery['artifact']['path']}")
                print("Captured areas only. Physical dimensions and completeness require review.")
                return
            job = build_session(Store(config.data_dir), session_id,
                                native_orientation_reviewed=args.reviewed_native_orientation,
                                quality=args.quality, fresh_baseline=args.fresh_baseline,
                                candidate=args.candidate,
                                frozen_source_job_id=args.frozen_source_job_id,
                                target_training_step=args.target_training_step,
                                incremental_new_captures=args.incremental_new_captures)
        except (ReconstructionError, StoreError, OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(2, f"Reconstruction unavailable or failed: {error}\n")
        print(f"Verified Gaussian PLY: {job['artifact']['path']}")
        if job.get('capture_sources') == ['android_camera2'] and not args.candidate:
            from .mesh import build_mesh
            try:
                mesh = build_mesh(Store(config.data_dir), job['project_id'], job['job_id'])
                print(f"Photo-derived triangle mesh: {mesh['artifact']['path']}")
            except (ReconstructionError, OSError, ValueError) as error:
                print(f"Mesh stage failed independently; completed splat retained: {error}")
        print("Quality candidate awaits review." if args.candidate else
              "Headset overlay availability is determined by the verified session preview endpoint.")
        return
    if args.export_session:
        if not args.export_output:
            parser.error("--export-output is required with --export-session")
        import uuid
        try:
            session_id = str(uuid.UUID(args.export_session))
        except ValueError:
            parser.error("--export-session must be a UUID")
        config = load_config(args.data_dir)
        manifest = Store(config.data_dir).export_session(session_id)
        output = args.export_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=output.parent, encoding="utf-8", delete=False) as handle:
            json.dump(manifest, handle, indent=2)
            temporary = Path(handle.name)
        os.replace(temporary, output)
        print(f"Exported {len(manifest['captures'])} captures to {output}")
        return
    if args.export_output:
        parser.error("--export-output requires --export-session")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback and (not args.tls_cert or not args.tls_key):
        parser.error("A non-loopback listener requires --tls-cert and --tls-key")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("Provide both --tls-cert and --tls-key")
    config = load_config(args.data_dir)
    print(f"Bridge data: {config.data_dir}")
    print(f"Bearer token file: {config.data_dir / 'bridge.token' if not os.environ.get('QUEST_SPLAT_TOKEN') else 'QUEST_SPLAT_TOKEN environment variable'}")
    dashboard_enabled = loopback and not args.tls_cert and args.port == 8765
    uvicorn.run(create_app(config, dashboard_enabled=dashboard_enabled), host=args.host, port=args.port,
                ssl_certfile=str(args.tls_cert) if args.tls_cert else None,
                ssl_keyfile=str(args.tls_key) if args.tls_key else None)


if __name__ == "__main__":
    main()
