import pickle, pathlib, gc, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data/results"

def inspect(name):
    path = RESULTS_DIR / f"{name}.pkl"
    with open(path, "rb") as fh:
        cached = pickle.load(fh)
    print("=" * 60)
    print(name, "data_version=", cached.get("data_version"))
    result = cached["result"]
    print("top-level type:", type(result))
    if isinstance(result, dict):
        print("subjects:", list(result.keys()))
        first_key = list(result.keys())[0]
        fold = result[first_key]
        print("fold keys:", list(fold.keys()))
        r = fold["result"]
        print("result dict keys:", list(r.keys()))
        for k, v in r.items():
            if k in ("model", "history", "run"):
                print(f"  {k}: <{type(v)}>")
            else:
                try:
                    print(f"  {k}: type={type(v)} shape/len={getattr(v,'shape',None) or len(v)}")
                except Exception as e:
                    print(f"  {k}: type={type(v)} (len failed: {e})")
        meta = fold["test_metadata"]
        print("test_metadata type:", type(meta), "shape:", meta.shape)
        print("test_metadata columns:", list(meta.columns))
        print("test_metadata EventType value_counts:")
        print(meta["EventType"].value_counts())
    del cached, result
    gc.collect()

if __name__ == "__main__":
    import sys
    inspect(sys.argv[1])
