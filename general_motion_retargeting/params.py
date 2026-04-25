import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

# Only Booster K1 and T1 are supported in this fork. Adding a robot means
# (a) a new dir under assets/, (b) a sole-points entry in sole_points.py,
# (c) a viewer cam distance, (d) the appropriate ik_configs/<src>_to_<robot>.json
# files, and (e) registering it in the four dicts below.

ROBOT_XML_DICT = {
    "booster_t1": ASSET_ROOT / "booster_t1" / "T1_serial.xml",
    "booster_k1": ASSET_ROOT / "booster_k1" / "K1_serial.xml",
}

IK_CONFIG_DICT = {
    "smplx": {
        "booster_t1": IK_CONFIG_ROOT / "smplx_to_t1.json",
        "booster_k1": IK_CONFIG_ROOT / "smplx_to_k1.json",
    },
    "bvh_lafan1": {
        "booster_t1": IK_CONFIG_ROOT / "bvh_lafan1_to_t1.json",
        "booster_k1": IK_CONFIG_ROOT / "bvh_lafan1_to_k1.json",
    },
}

ROBOT_BASE_DICT = {
    "booster_t1": "Waist",
    "booster_k1": "Trunk",
}

VIEWER_CAM_DISTANCE_DICT = {
    "booster_t1": 2.0,
    "booster_k1": 2.0,
}
