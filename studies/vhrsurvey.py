# SPDX-License-Identifier: MIT
import os, glob, numpy as np, vhr
rows = [("file", "dims", "voxels", "freqs", "ports", "status")]
for f in sorted(glob.glob('VoxHenry/Input_files/*.vhr')):
    b = os.path.basename(f)
    try:
        m = vhr.read_vhr(f)
    except Exception as e:
        rows.append((b, "PARSE FAIL", type(e).__name__, str(e)[:40], "", ""))
        continue
    d = np.asarray(m.dims, dtype=int)
    nvox = int(np.count_nonzero(m.struc()))
    fq = np.atleast_1d(np.asarray(m.freq, dtype=float))
    frs = ("%d (%.3g)" % (fq.size, fq[0]) if fq.size == 1
           else "%d (%.3g..%.3g)" % (fq.size, fq.min(), fq.max()))
    blk = []
    if m.superconductor:
        blk.append("SUPERCOND: no London model")
    try:
        m.uniform_sigma()
    except Exception as e:
        blk.append("MIXED SIGMA")
    rows.append((b, "%dx%dx%d" % tuple(d), "%d (%.1f%%)" % (nvox, m.fill_pct()), frs,
                 str(len(m.ports)), "; ".join(blk) or "buildable"))
w = [max(len(str(r[i])) for r in rows) for i in range(6)]
for r in rows:
    print("  ".join(str(r[i]).ljust(w[i]) for i in range(6)))
