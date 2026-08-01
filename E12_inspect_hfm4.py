"""
E12 step 1: inspect HuggingFaceM4/SugarCrepe's actual schema before writing any
probe/heuristic code against it.

We've been using haideraltahan/wds_sugarcrepe throughout (confirmed to be the
exact dataset the original paper's own scripts use). The user wants to
cross-check against HuggingFaceM4/SugarCrepe directly, in case its packaging
differs in a way that matters. Rather than assume field names, this script
just loads it and prints what's actually there: available configs (SugarCrepe
is organized as one config per category upstream, e.g. swap_obj, add_obj),
the feature schema, and a couple of raw examples per config.

Do not write any probe code against this until we've seen this output --
this dataset is known to only carry the NEGATIVE caption plus a COCO image
reference in its original (non-webdataset) form, not both captions directly,
which would change how "true caption" has to be recovered.

Output: prints only, plus E12_schema.json
"""

import json

from datasets import get_dataset_config_names, load_dataset

REPO = "HuggingFaceM4/SugarCrepe"


def main():
    print(f"Fetching config names for {REPO} ...")
    configs = get_dataset_config_names(REPO)
    print("configs:", configs)

    schema = {"repo": REPO, "configs": {}}

    for cfg in configs:
        print(f"\n=== config: {cfg} ===")
        ds = load_dataset(REPO, cfg)
        print("splits:", list(ds.keys()))
        split_name = list(ds.keys())[0]
        split = ds[split_name]
        print("features:", split.features)
        print("n_examples:", len(split))
        examples = [split[i] for i in range(min(2, len(split)))]
        print("first 2 raw examples:")
        for ex in examples:
            printable = {k: (v if not hasattr(v, "size") else f"<image {getattr(v, 'size', '?')}>")
                         for k, v in ex.items()}
            print(json.dumps(printable, indent=2, default=str))

        schema["configs"][cfg] = {
            "splits": list(ds.keys()),
            "n_examples": len(split),
            "features": {k: str(v) for k, v in split.features.items()},
            "example_keys": list(examples[0].keys()) if examples else [],
        }

    with open("E12_schema.json", "w") as f:
        json.dump(schema, f, indent=2, default=str)
    print("\nwrote E12_schema.json")


if __name__ == "__main__":
    main()
