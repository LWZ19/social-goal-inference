import pickle, collections
P = pickle.load(open("results/still-scenes/facing/verify/v_pairs.pkl", "rb"))["pairs"]
def view(a): return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"
for s in ["qwen_s19", "gemini"]:
    rows = [r for r in P if r["src"] == s]
    print(s, "line by view", dict(collections.Counter(view(r["ang"]) for r in rows if r["role"] == "line")),
          "| profile by role", dict(collections.Counter(r["role"] for r in rows if view(r["ang"]) == "profile")))
