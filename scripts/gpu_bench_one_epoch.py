#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.34"]
# ///
"""
Bench one epoch of YOLO training on an AWS GPU spot instance.

Stages a clean copy of dataset/ locally (resolving symlinks, skipping
broken ones), uploads via presigned S3 URLs, launches a g4dn.xlarge
spot instance with a safety-net shutdown, runs N epochs, fetches the
timing, and prints actual + projected full-run cost. Instance is
terminated on every exit path.

No IAM instance profile needed — the instance accesses S3 only via
presigned URLs baked into user-data.

Usage:
  export S3_BUCKET=my-bucket
  ./scripts/gpu_bench_one_epoch.py
  ./scripts/gpu_bench_one_epoch.py --dry-run          # stage only
  ./scripts/gpu_bench_one_epoch.py --epochs 2         # more signal
  ./scripts/gpu_bench_one_epoch.py --model yolo26s.pt # override model

Required: AWS credentials with ec2:* and s3:* on the target bucket.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


DEFAULTS = {
    "region": "us-east-1",
    "instance_type": "g4dn.xlarge",
    "model": "yolo26n.pt",
    "spot_max_price": "0.25",
    "safety_min": 45,
    "projected_epochs": 40,
    "imgsz": 416,
    "batch": 16,
    "ebs_gb": 100,
    "pricing": "spot",
    "on_demand_price": 0.526,  # g4dn.xlarge us-east-1 on-demand $/hr (Apr 2026)
}


def log(msg):
    print(f"[bench] {msg}", flush=True)


def stage_dataset(src: Path, stage: Path, model_file: Path,
                  include_augmented: bool) -> dict:
    """Copy real image+label pairs into stage/.

    Labels in dataset/labels/{train,val}/ are the source of truth. Images are
    located by stem: `aug_*` → dataset/augmented/, otherwise → scanmyphotos/.
    Augmented images are skipped unless include_augmented=True.
    """
    shutil.copy2(model_file, stage / model_file.name)
    (stage / "dataset").mkdir(parents=True)

    scanmyphotos = Path("scanmyphotos")
    augmented = src / "augmented"
    if not scanmyphotos.is_dir():
        raise RuntimeError("scanmyphotos/ not found — can't locate source images")

    stats = {"train": 0, "val": 0, "skipped_aug": 0, "skipped_no_image": 0}
    for split in ("train", "val"):
        lbl_src = src / "labels" / split
        img_dst = stage / "dataset" / "images" / split
        lbl_dst = stage / "dataset" / "labels" / split
        img_dst.mkdir(parents=True)
        lbl_dst.mkdir(parents=True)

        if not lbl_src.is_dir():
            continue
        for lbl in sorted(lbl_src.iterdir()):
            if lbl.suffix != ".txt":
                continue
            stem = lbl.stem
            is_aug = stem.startswith("aug_")
            if is_aug and not include_augmented:
                stats["skipped_aug"] += 1
                continue
            img_src = (augmented if is_aug else scanmyphotos) / f"{stem}.jpg"
            if not img_src.is_file():
                stats["skipped_no_image"] += 1
                continue
            shutil.copy2(img_src, img_dst / img_src.name)
            shutil.copy2(lbl, lbl_dst / lbl.name)
            stats[split] += 1

    (stage / "dataset" / "data.yaml").write_text(
        "names:\n  0: target\n"
        "path: /app/dataset\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    return stats


def make_tar(stage: Path, out: Path):
    subprocess.run(
        ["tar", "--zstd", "-C", str(stage), "-cf", str(out), "."],
        check=True,
    )


def get_ami_min_volume_gb(ec2, ami_id: str) -> int:
    """Return the largest EBS volume size declared in the AMI's block device mappings."""
    resp = ec2.describe_images(ImageIds=[ami_id])
    if not resp["Images"]:
        return 0
    sizes = [
        bdm["Ebs"]["VolumeSize"]
        for bdm in resp["Images"][0].get("BlockDeviceMappings", [])
        if "Ebs" in bdm and bdm["Ebs"].get("VolumeSize")
    ]
    return max(sizes) if sizes else 0


def get_latest_dlami(ec2) -> str:
    patterns = [
        "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*",
        "Deep Learning Base GPU AMI (Ubuntu 22.04)*",
    ]
    for pat in patterns:
        resp = ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": [pat]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        if resp["Images"]:
            return sorted(resp["Images"], key=lambda i: i["CreationDate"])[-1]["ImageId"]
    raise RuntimeError("no matching Deep Learning Base AMI found in region")


def fetch_spot_price(ec2, instance_type: str) -> float:
    resp = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        MaxResults=1,
    )
    if not resp["SpotPriceHistory"]:
        raise RuntimeError(f"no spot price history for {instance_type}")
    return float(resp["SpotPriceHistory"][0]["SpotPrice"])


def resolve_network(ec2) -> list[tuple[str, str, str]]:
    """Find all (subnet_id, security_group_id, az) candidates for public subnets.

    Returns a list so callers can retry across AZs on InsufficientInstanceCapacity.
    Prefers default VPC; falls back to any VPC with an attached IGW.
    """
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_ids = [v["VpcId"] for v in vpcs["Vpcs"]]
    if not vpc_ids:
        igws = ec2.describe_internet_gateways()
        vpc_ids = [
            att["VpcId"]
            for igw in igws["InternetGateways"]
            for att in igw["Attachments"]
            if att.get("State") == "available"
        ]
    if not vpc_ids:
        raise RuntimeError("no VPC with internet access found in this region")

    subnets = ec2.describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": vpc_ids},
        {"Name": "state", "Values": ["available"]},
    ])
    public = [s for s in subnets["Subnets"] if s.get("MapPublicIpOnLaunch")]
    if not public:
        raise RuntimeError("no public subnet (MapPublicIpOnLaunch=True) available")

    # Cache SGs per VPC to avoid repeated calls
    sg_cache: dict[str, str] = {}
    candidates: list[tuple[str, str, str]] = []
    for subnet in public:
        vpc_id = subnet["VpcId"]
        if vpc_id not in sg_cache:
            sgs = ec2.describe_security_groups(Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "group-name", "Values": ["default"]},
            ])
            if not sgs["SecurityGroups"]:
                continue
            sg_cache[vpc_id] = sgs["SecurityGroups"][0]["GroupId"]
        candidates.append((subnet["SubnetId"], sg_cache[vpc_id], subnet["AvailabilityZone"]))
    if not candidates:
        raise RuntimeError("no public subnets with a default security group")
    return candidates


def build_user_data(*, payload_get, timing_put, log_put, run_put, model, imgsz,
                    batch, epochs, safety_min) -> str:
    train_timeout = safety_min * 60 - 300
    wall_safety = safety_min * 60
    return f"""#!/bin/bash
set -x
exec > /var/log/user-data.log 2>&1

# belt-and-suspenders: wall-clock shutdown regardless of script state
( sleep {wall_safety} && shutdown -h now ) &

upload_log() {{
  curl -sS --retry 3 -X PUT --upload-file /var/log/user-data.log "{log_put}" || true
}}
trap upload_log EXIT

mkdir -p /workspace && cd /workspace
curl -fsSL --retry 3 -o payload.tar.zst "{payload_get}"
tar --zstd -xf payload.tar.zst
rm payload.tar.zst

cat > /workspace/run.py <<'PYEOF'
import json, time
from ultralytics import YOLO
t0 = time.time()
model = YOLO("/workspace/{model}")
results = model.train(
    data="/app/dataset/data.yaml",
    epochs={epochs},
    imgsz={imgsz},
    batch={batch},
    device=0,
    workers=4,
    project="/workspace/runs",
    name="bench",
    exist_ok=True,
    patience=0,
    cache=False,
    verbose=True,
)
elapsed = time.time() - t0
out = {{"elapsed_sec": elapsed, "epochs": {epochs}}}
try:
    rd = getattr(results, "results_dict", None) or {{}}
    out["results_dict"] = {{str(k): (float(v) if hasattr(v, "__float__") else str(v)) for k, v in rd.items()}}
except Exception as e:
    out["results_dict_error"] = repr(e)
try:
    m = getattr(model, "metrics", None)
    if m is not None and getattr(m, "box", None) is not None:
        out["val_map50"] = float(m.box.map50)
        out["val_map50_95"] = float(m.box.map)
        out["val_precision"] = float(m.box.mp)
        out["val_recall"] = float(m.box.mr)
except Exception as e:
    out["val_metrics_error"] = repr(e)
with open("/workspace/timing.json", "w") as f:
    json.dump(out, f)
print("ELAPSED_SEC", elapsed)
print("METRICS", json.dumps(out))
PYEOF

docker pull ultralytics/ultralytics:latest
timeout {train_timeout} docker run --rm --gpus all \\
  -v /workspace:/workspace \\
  -v /workspace/dataset:/app/dataset \\
  -w /workspace \\
  ultralytics/ultralytics:latest \\
  python /workspace/run.py

if [[ -d /workspace/runs/bench ]]; then
  tar --zstd -cf /workspace/run.tar.zst -C /workspace/runs bench
  curl -sS --retry 3 -X PUT --upload-file /workspace/run.tar.zst "{run_put}" || true
fi

if [[ -f /workspace/timing.json ]]; then
  curl -sS --retry 3 -X PUT --upload-file /workspace/timing.json "{timing_put}"
fi

shutdown -h now
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULTS["region"]))
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"),
                    help="S3 bucket for payload transfer (env: S3_BUCKET)")
    ap.add_argument("--instance-type", default=DEFAULTS["instance_type"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--epochs", type=int, default=1, help="epochs to time")
    ap.add_argument("--spot-max-price", default=DEFAULTS["spot_max_price"])
    ap.add_argument("--safety-min", type=int, default=DEFAULTS["safety_min"],
                    help="hard wall-clock shutdown after this many minutes")
    ap.add_argument("--projected-epochs", type=int, default=DEFAULTS["projected_epochs"],
                    help="extrapolate full-run cost to this many epochs")
    ap.add_argument("--ebs-gb", type=int, default=DEFAULTS["ebs_gb"])
    ap.add_argument("--pricing", choices=["spot", "on-demand"], default=DEFAULTS["pricing"],
                    help="EC2 pricing model (default: spot, requires spot quota)")
    ap.add_argument("--on-demand-price", type=float, default=DEFAULTS["on_demand_price"],
                    help="$/hr for cost math when --pricing=on-demand (g4dn.xlarge us-east-1)")
    ap.add_argument("--include-augmented", action="store_true",
                    help="include 15k augmented images (7x slower; off by default)")
    ap.add_argument("--dry-run", action="store_true", help="stage + package only, no AWS calls")
    ap.add_argument("--keep-artifacts", action="store_true", help="don't delete S3 artifacts after")
    args = ap.parse_args()

    if not args.dry_run and not args.bucket:
        sys.exit("set S3_BUCKET env var or pass --bucket")
    if not Path(args.model).is_file():
        sys.exit(f"model file not found: {args.model} (cwd: {Path.cwd()})")
    if not Path("dataset").is_dir():
        sys.exit("no dataset/ here — run from repo root")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    s3_prefix = f"yolo-gpu-runs/{run_id}"
    log(f"run id: {run_id}")

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        stage = td / "stage"
        stage.mkdir()

        log(f"staging dataset (augmented={'on' if args.include_augmented else 'off'})...")
        stats = stage_dataset(Path("dataset"), stage, Path(args.model), args.include_augmented)
        log(f"  train={stats['train']}  val={stats['val']}  "
            f"skipped_aug={stats['skipped_aug']}  "
            f"skipped_no_image={stats['skipped_no_image']}")
        if stats["train"] == 0 or stats["val"] == 0:
            sys.exit("empty train or val split after staging — check source images")

        tarball = td / "payload.tar.zst"
        log("packaging with zstd...")
        make_tar(stage, tarball)
        size_mb = tarball.stat().st_size / 1024 / 1024
        log(f"  payload: {size_mb:.1f} MB")

        if args.dry_run:
            log("dry-run: skipping AWS")
            return

        session = boto3.Session(region_name=args.region)
        ec2 = session.client("ec2")
        s3 = session.client("s3")

        # AWS preflight — fail BEFORE uploading 1.8GB so retries are cheap
        log("resolving latest Deep Learning Base AMI...")
        ami_id = get_latest_dlami(ec2)
        ami_min_gb = get_ami_min_volume_gb(ec2, ami_id)
        volume_gb = max(args.ebs_gb, ami_min_gb)
        log(f"  ami: {ami_id}  ami_min_root={ami_min_gb}GB  using={volume_gb}GB")

        log("resolving network (subnet + security group)...")
        candidates = resolve_network(ec2)
        log(f"  {len(candidates)} public subnet candidate(s): "
            + ", ".join(f"{az}:{sid}" for sid, _, az in candidates))

        if args.pricing == "spot":
            hourly_rate = fetch_spot_price(ec2, args.instance_type)
            log(f"  spot price: ${hourly_rate:.4f}/hr  (cap: ${args.spot_max_price})")
            if hourly_rate > float(args.spot_max_price):
                sys.exit(f"current spot price ${hourly_rate} exceeds cap ${args.spot_max_price} — "
                         f"raise --spot-max-price or try another region/AZ")
        else:
            hourly_rate = args.on_demand_price
            log(f"  on-demand price: ${hourly_rate:.4f}/hr (assumed; not fetched)")

        # Upload only after preflight passes
        log(f"uploading to s3://{args.bucket}/{s3_prefix}/payload.tar.zst ...")
        payload_key = f"{s3_prefix}/payload.tar.zst"
        timing_key = f"{s3_prefix}/timing.json"
        log_key = f"{s3_prefix}/user-data.log"
        run_key = f"{s3_prefix}/run.tar.zst"
        s3.upload_file(str(tarball), args.bucket, payload_key)

        expiry = max(4 * 3600, args.safety_min * 60 + 900)
        payload_get = s3.generate_presigned_url(
            "get_object", Params={"Bucket": args.bucket, "Key": payload_key}, ExpiresIn=expiry)
        timing_put = s3.generate_presigned_url(
            "put_object", Params={"Bucket": args.bucket, "Key": timing_key}, ExpiresIn=expiry)
        log_put = s3.generate_presigned_url(
            "put_object", Params={"Bucket": args.bucket, "Key": log_key}, ExpiresIn=expiry)
        run_put = s3.generate_presigned_url(
            "put_object", Params={"Bucket": args.bucket, "Key": run_key}, ExpiresIn=expiry)

        user_data = build_user_data(
            payload_get=payload_get, timing_put=timing_put, log_put=log_put,
            run_put=run_put,
            model=Path(args.model).name, imgsz=args.imgsz, batch=args.batch,
            epochs=args.epochs, safety_min=args.safety_min,
        )

        log(f"launching {args.pricing} instance...")
        base_kwargs = dict(
            ImageId=ami_id,
            InstanceType=args.instance_type,
            MinCount=1, MaxCount=1,
            UserData=user_data,
            InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_gb, "VolumeType": "gp3",
                        "DeleteOnTermination": True},
            }],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"yolo-bench-{run_id}"},
                    {"Key": "RunId", "Value": run_id},
                    {"Key": "Purpose", "Value": "yolo-bench-one-epoch"},
                ],
            }],
        )
        if args.pricing == "spot":
            base_kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "MaxPrice": args.spot_max_price,
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }
        resp = None
        last_err = None
        for subnet_id, sg_id, az in candidates:
            try:
                resp = ec2.run_instances(
                    SubnetId=subnet_id, SecurityGroupIds=[sg_id], **base_kwargs,
                )
                log(f"  launched in {az} ({subnet_id})")
                break
            except ClientError as e:
                code = e.response["Error"]["Code"]
                last_err = (code, str(e))
                if code == "InsufficientInstanceCapacity":
                    log(f"  no capacity in {az}, trying next AZ...")
                    continue
                if code == "Unsupported":
                    log(f"  {args.instance_type} unsupported in {az}, trying next AZ...")
                    continue
                if code in ("SpotMaxPriceTooLow", "MaxSpotInstanceCountExceeded",
                            "VcpuLimitExceeded"):
                    sys.exit(f"launch failed ({code}) — raise --spot-max-price, "
                             f"switch --pricing, or check vCPU quota")
                raise
        if resp is None:
            sys.exit(f"launch failed after exhausting {len(candidates)} AZs; "
                     f"last error: {last_err}")

        instance_id = resp["Instances"][0]["InstanceId"]
        launch_time = time.time()
        log(f"  instance: {instance_id}")
        log(f"  dashboard: https://{args.region}.console.aws.amazon.com/ec2/home"
            f"?region={args.region}#InstanceDetails:instanceId={instance_id}")

        timing = None
        try:
            deadline = launch_time + args.safety_min * 60
            log(f"waiting for timing.json in s3 (up to {args.safety_min}m)...")
            poll = 0
            while time.time() < deadline:
                poll += 1
                try:
                    s3.head_object(Bucket=args.bucket, Key=timing_key)
                    log("  timing.json ready")
                    break
                except ClientError as e:
                    if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "Not Found"):
                        raise
                try:
                    ins = ec2.describe_instances(InstanceIds=[instance_id])
                    state = ins["Reservations"][0]["Instances"][0]["State"]["Name"]
                except ClientError:
                    state = "unknown"
                if state == "terminated":
                    log("  instance terminated before timing.json appeared")
                    try:
                        s3.download_file(args.bucket, log_key, "/tmp/yolo-bench-user-data.log")
                        log("  user-data log: /tmp/yolo-bench-user-data.log")
                    except ClientError:
                        log("  no user-data log in s3 (instance may have died early)")
                    sys.exit(1)
                if poll % 6 == 0:
                    mins = int((time.time() - launch_time) / 60)
                    log(f"  [{mins}m] still waiting, instance={state}")
                time.sleep(20)
            else:
                log("deadline reached without result")
                sys.exit(1)

            s3.download_file(args.bucket, timing_key, str(td / "timing.json"))
            timing = json.loads((td / "timing.json").read_text())

            run_tar = td / "run.tar.zst"
            try:
                s3.download_file(args.bucket, run_key, str(run_tar))
                out_dir = Path("runs/detect/gpu-40ep")
                out_dir.mkdir(parents=True, exist_ok=True)
                import subprocess
                subprocess.run(
                    ["tar", "--zstd", "-xf", str(run_tar), "-C", str(out_dir)],
                    check=True,
                )
                log(f"  run artifacts (weights + tensorboard) extracted to {out_dir}")
            except ClientError as e:
                log(f"  WARN: run artifacts download failed: {e}")
        finally:
            try:
                log(f"terminating {instance_id}...")
                ec2.terminate_instances(InstanceIds=[instance_id])
            except ClientError as e:
                log(f"  WARNING terminate failed: {e}")
                log(f"  MANUALLY TERMINATE: aws ec2 terminate-instances "
                    f"--region {args.region} --instance-ids {instance_id}")

            if not args.keep_artifacts:
                for key in (payload_key, timing_key, log_key, run_key):
                    try:
                        s3.delete_object(Bucket=args.bucket, Key=key)
                    except ClientError:
                        pass

        # === cost report ===
        elapsed = float(timing["elapsed_sec"])
        wall = time.time() - launch_time
        per_epoch = elapsed / args.epochs
        epoch_cost = per_epoch / 3600 * hourly_rate
        billed_cost = wall / 3600 * hourly_rate
        proj_cost = epoch_cost * args.projected_epochs
        proj_min = per_epoch * args.projected_epochs / 60

        print()
        print("=" * 56)
        print("  GPU BENCH RESULTS")
        print("=" * 56)
        print(f"  instance type:          {args.instance_type}  ({args.pricing})")
        print(f"  rate at launch:         ${hourly_rate:.4f}/hr")
        print(f"  epochs measured:        {args.epochs}")
        print(f"  training time:          {elapsed:.1f} s  ({elapsed/60:.2f} min)")
        print(f"  per-epoch time:         {per_epoch:.1f} s  ({per_epoch/60:.2f} min)")
        print(f"  wall-clock (billed):    {wall:.1f} s  ({wall/60:.2f} min)")
        print("  " + "-" * 52)
        print(f"  cost of this bench run: ${billed_cost:.4f}")
        print(f"  cost per epoch:         ${epoch_cost:.4f}")
        print("  " + "-" * 52)
        print(f"  projected {args.projected_epochs}-epoch run (training only):")
        print(f"    wall time:              {proj_min:.0f} min")
        print(f"    training cost:          ${proj_cost:.4f}")
        print(f"    +15% boot/pull overhead:${proj_cost * 1.15:.4f}")
        print("=" * 56)
        if timing.get("val_map50") is not None:
            print("  validation metrics (after 1 epoch):")
            print(f"    mAP50:       {timing['val_map50']:.4f}")
            print(f"    mAP50-95:    {timing['val_map50_95']:.4f}")
            print(f"    precision:   {timing['val_precision']:.4f}")
            print(f"    recall:      {timing['val_recall']:.4f}")
        elif timing.get("val_metrics_error"):
            print(f"  val metrics capture failed: {timing['val_metrics_error']}")
        if timing.get("results_dict"):
            print("  training losses:")
            for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
                if k in timing["results_dict"]:
                    print(f"    {k:18s}: {timing['results_dict'][k]}")
        print("=" * 56)
        print()
        print("caveats:")
        print("  - first epoch includes model compile/warmup; real per-epoch")
        print("    in a longer run is typically 5-20% faster")
        print("  - patience=10 usually early-stops well before 40 epochs")
        print("    (model hit mAP50=0.95 after epoch 1 on CPU); halve again")
        print("  - spot price is a snapshot; actual billed rate can drift")


if __name__ == "__main__":
    main()
