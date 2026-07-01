// Pose-agnostic banded dynamic time warping.
//
// Aligns a query sequence (rows, 0..n-1) against a reference sequence
// (columns, 0..m-1) allowing local nonlinear time warping within a bounded
// band around a caller-supplied anchor per row. Both the sequence start and
// end are "open" (not pinned), so the path may begin/end wherever is
// cheapest inside the band - this is what gives real temporal "room" instead
// of a rigid one-to-one frame lookup or a handful of discrete global speeds.

/**
 * @param {object} params
 * @param {number} params.n - query length (live samples)
 * @param {number} params.m - reference length (dance frames)
 * @param {(i: number, j: number) => number} params.cost - cost in ~[0, 1] for aligning row i to column j
 * @param {(i: number) => number} params.anchorForRow - expected reference index for row i
 * @param {number} params.bandRadius - +/- band width (in reference-index units) around the anchor
 * @param {number} [params.stepPenalty] - extra cost added for non-diagonal (insertion/deletion) steps
 * @returns {{ path: [number, number][], meanCost: number, alignedIndex: number|null, startIndex: number|null }}
 */
export function bandedDtw({ n, m, cost, anchorForRow, bandRadius, stepPenalty = 0 }) {
  if (n <= 0 || m <= 0) {
    return { path: [], meanCost: 1, alignedIndex: null, startIndex: null };
  }

  const bandLow = new Int32Array(n);
  const bandHigh = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const center = anchorForRow(i);
    bandLow[i] = Math.max(0, Math.floor(center - bandRadius));
    bandHigh[i] = Math.min(m - 1, Math.ceil(center + bandRadius));
    if (bandHigh[i] < bandLow[i]) bandHigh[i] = bandLow[i];
  }

  const D = new Array(n);
  const rowParents = new Array(n);

  for (let i = 0; i < n; i++) {
    const lo = bandLow[i];
    const hi = bandHigh[i];
    const width = hi - lo + 1;
    const row = new Float64Array(width);
    const parents = new Int8Array(width);

    const prevLo = i > 0 ? bandLow[i - 1] : 0;
    const prevHi = i > 0 ? bandHigh[i - 1] : -1;
    const prevRow = i > 0 ? D[i - 1] : null;

    for (let j = lo; j <= hi; j++) {
      const c = cost(i, j);
      let best = Infinity;
      let dir = -1; // -1 = open start (no predecessor)

      if (i > 0) {
        if (j - 1 >= prevLo && j - 1 <= prevHi) {
          const v = prevRow[j - 1 - prevLo];
          if (v < best) { best = v; dir = 0; } // diagonal
        }
        if (j >= prevLo && j <= prevHi) {
          const v = prevRow[j - prevLo] + stepPenalty;
          if (v < best) { best = v; dir = 1; } // vertical (live advances, ref held)
        }
        if (j - 1 >= lo) {
          const v = row[j - 1 - lo] + stepPenalty;
          if (v < best) { best = v; dir = 2; } // horizontal (ref advances, live held)
        }
      }

      if (dir === -1) {
        best = c;
      } else {
        best += c;
      }

      row[j - lo] = best;
      parents[j - lo] = dir;
    }

    D[i] = row;
    rowParents[i] = parents;
  }

  const lastLo = bandLow[n - 1];
  const lastHi = bandHigh[n - 1];
  const lastRow = D[n - 1];
  let bestJ = lastLo;
  let bestCost = Infinity;
  for (let j = lastLo; j <= lastHi; j++) {
    const v = lastRow[j - lastLo];
    if (v < bestCost) {
      bestCost = v;
      bestJ = j;
    }
  }

  const path = [];
  let i = n - 1;
  let j = bestJ;
  for (;;) {
    path.push([i, j]);
    const lo = bandLow[i];
    const dir = rowParents[i][j - lo];
    if (dir === -1) break;
    if (dir === 0) { i -= 1; j -= 1; }
    else if (dir === 1) { i -= 1; }
    else { j -= 1; }
  }
  path.reverse();

  return {
    path,
    meanCost: path.length > 0 ? bestCost / path.length : 1,
    alignedIndex: bestJ,
    startIndex: path[0]?.[1] ?? null,
  };
}
