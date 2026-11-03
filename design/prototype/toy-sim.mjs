// Reference toy simulation extracted from twinlight-city.html (same code, same seed).
// The Python engine ports this in P-010 to P-013. Run: node design/prototype/toy-sim.mjs [residents]
const HOURS = 720;
const RAIN_A = 264, RAIN_B = 336;
const AD_HOUR = 9, CREATOR_HOUR = 186, RESTOCK_HOUR = 14;
const DELIVERY_R = 900;
const SPILL_R = 1500;
const STORE_N = 14;
const CITY = { cx: -250, cz: 0, rx: 3500, rz: 3200 };
const OLD_C = [1400, 150], CBD_C = [-1500, -350];
const DISTRICTS = [
  ['Ellisbridge', -250, 300], ['Paldi', -350, 1450], ['Navrangpura', -950, -550],
  ['Naranpura', -850, -1650], ['Chandkheda', -250, -2650], ['Ghatlodia', -1750, -2250],
  ['Vastrapur', -2000, 350], ['Bodakdev', -2350, -900], ['Thaltej', -3050, -1500],
  ['Satellite', -1750, 1500], ['Shahibaug', 1650, -1350], ['Maninagar', 1850, 1350],
].map(([name, x, z], i) => ({ name, x, z, i, east: x > riverX(z) }));
const CREATOR_DISTRICTS = ['Satellite', 'Bodakdev', 'Vastrapur'];
const FIRST = ['Ritu', 'Aarav', 'Diya', 'Krish', 'Isha', 'Vihaan', 'Anaya', 'Kavya', 'Rohan', 'Priya', 'Neel', 'Hetal', 'Jignesh', 'Bhavna', 'Nirav', 'Foram', 'Dhruv', 'Mansi', 'Kunal', 'Pooja', 'Harsh', 'Riya', 'Yash', 'Nidhi', 'Parth', 'Khushi', 'Meet', 'Janvi', 'Tanvi', 'Chirag', 'Sneha', 'Aditya', 'Megha', 'Hemant', 'Komal', 'Rajesh', 'Sunita', 'Mahesh', 'Geeta', 'Darshan', 'Urvi', 'Ketan', 'Payal', 'Tejas', 'Avni', 'Mihir', 'Shruti', 'Bhavesh'];
const LAST = ['Patel', 'Shah', 'Desai', 'Mehta', 'Joshi', 'Trivedi', 'Parikh', 'Bhatt', 'Modi', 'Pandya', 'Vyas', 'Thakkar', 'Chauhan', 'Rana', 'Solanki', 'Parmar', 'Dave', 'Raval', 'Panchal', 'Gandhi', 'Amin', 'Kapadia', 'Sheth', 'Dalal', 'Oza', 'Brahmbhatt', 'Soni', 'Contractor'];
const AGE_BANDS = [[16, 24], [25, 34], [35, 49], [50, 64], [65, 80]];
const AGE_P_WEST = [0.2, 0.27, 0.27, 0.16, 0.1];
const AGE_P_EAST = [0.16, 0.22, 0.28, 0.21, 0.13];
const AGE_AFF = [1.35, 1.2, 0.9, 0.6, 0.38];
const AGE_AD = [1.0, 0.9, 0.62, 0.38, 0.2];
const TOD = [0.25, 0.12, 0.06, 0.04, 0.04, 0.06, 0.15, 0.3, 0.4, 0.45, 0.5, 0.6, 0.75, 0.8, 0.65, 0.6, 0.7, 0.85, 1.0, 1.35, 1.55, 1.45, 1.1, 0.6];
const TALK = [0.35, 0.15, 0.06, 0.04, 0.04, 0.08, 0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0, 1.0, 0.9, 0.85, 0.9, 1.0, 1.15, 1.3, 1.35, 1.25, 1.0, 0.6];
const P = {
  adDay1: 0.14, adDrip: 0.003, creatorIn: 0.3, creatorOut: 0.03,
  betaA: 0.04, betaW: 0.01, a0: 0.0051, r0: 0.0075,
  stockInit: 0.036, stockPlan: 0.018, stockBase: 0.006, restockK: 1.05, capK: 0.04, capUntil: 240, safety: 1.3,
};

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hash3(a, b, c) {
  let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263) + Math.imul(c | 0, 1274126177)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1103515245);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
function riverX(z) { return 520 + 230 * Math.sin(z / 1250 + 0.6) + 110 * Math.sin(z / 470 + 2.1); }
function riverHW(z) { return 88 + 18 * Math.sin(z / 800 + 1.0); }
function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function cityF(x, z) {
  const dx = (x - CITY.cx) / CITY.rx, dz = (z - CITY.cz) / CITY.rz;
  const th = Math.atan2(dz, dx);
  const r = 1 + 0.07 * Math.sin(3 * th + 1) + 0.05 * Math.sin(5 * th + 2) + 0.03 * Math.sin(9 * th);
  return Math.hypot(dx, dz) / r;
}
function peakAt(x, z) {
  const g = (cx, cz, s) => Math.exp(-((x - cx) ** 2 + (z - cz) ** 2) / (2 * s * s));
  return Math.max(g(CBD_C[0], CBD_C[1], 850), 0.62 * g(OLD_C[0], OLD_C[1], 650), 0.5 * g(-2650, -1350, 520), 0.36 * g(-1750, 1500, 520));
}
const SIDE = [DISTRICTS.filter(d => !d.east), DISTRICTS.filter(d => d.east)];
function districtAt(x, z) {
  let best = -1, bd = Infinity;
  for (const d of SIDE[x > riverX(z) ? 1 : 0]) {
    const dd = (x - d.x) ** 2 + (z - d.z) ** 2;
    if (dd < bd) { bd = dd; best = d.i; }
  }
  return best;
}

function arterialLines() {
  const sample = (fn, a, b, n) => Array.from({ length: n + 1 }, (_, k) => fn(a + (b - a) * k / n));
  const ring = sample(a => {
    const w = 1 + 0.03 * Math.sin(a * 5 + 0.7);
    return [-250 + 2700 * w * Math.cos(a), -50 + 2450 * w * Math.sin(a)];
  }, 0, Math.PI * 2, 220);
  const diag = (p, q) => sample(t => [p[0] + (q[0] - p[0]) * t + 70 * Math.sin(t * 5.5), p[1] + (q[1] - p[1]) * t], 0, 1, 70);
  return [
    ring,
    sample(z => [-2750 + 180 * Math.sin(z / 1600), z], -3900, 3700, 130),
    sample(z => [riverX(z) - riverHW(z) - 150, z], -3400, 3000, 130),
    sample(x => [x, -400 + 120 * Math.sin(x / 900)], -4000, 3400, 130),
    sample(x => [x, 1350 + 100 * Math.sin(x / 1100 + 1)], -3800, 3200, 130),
    sample(x => [x, -1750 + 90 * Math.sin(x / 700)], -3700, 3100, 130),
    diag(OLD_C, [3300, -2600]), diag(OLD_C, [3200, 2800]), diag(OLD_C, [1000, -3400]),
    diag(CBD_C, [-3600, -3100]), diag(CBD_C, [-3500, 2700]),
  ];
}

function buildCity(rand) {
  const G = 20, GO = 4800, GN = 480;
  const artMask = new Uint8Array(GN * GN);
  const cellOf = (x, z) => {
    const gx = Math.floor((x + GO) / G), gz = Math.floor((z + GO) / G);
    return gx < 0 || gz < 0 || gx >= GN || gz >= GN ? -1 : gz * GN + gx;
  };
  const mark = (x, z) => {
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
      const c = cellOf(x + a * G, z + b * G);
      if (c >= 0) artMask[c] = 1;
    }
  };
  const onArt = (x, z) => { const c = cellOf(x, z); return c >= 0 && artMask[c] === 1; };
  const inRiver = (x, z, pad) => Math.abs(x - riverX(z)) < riverHW(z) + pad;

  const artSeg = [];
  for (const line of arterialLines()) {
    for (let k = 0; k < line.length - 1; k++) {
      const [x1, z1] = line[k], [x2, z2] = line[k + 1];
      if (cityF((x1 + x2) / 2, (z1 + z2) / 2) > 1.08) continue;
      artSeg.push(x1, z1, x2, z2);
      const n = Math.ceil(Math.hypot(x2 - x1, z2 - z1) / 10);
      for (let q = 0; q <= n; q++) mark(x1 + (x2 - x1) * q / n, z1 + (z2 - z1) * q / n);
    }
  }

  const B = { x: [], z: [], w: [], d: [], h: [], rot: [], dist: [] };
  const roadSeg = [], corners = [];

  const fillBlock = (c00, c10, c11, c01, s, f, di) => {
    const bil = (u, v) => [
      (1 - u) * (1 - v) * c00[0] + u * (1 - v) * c10[0] + u * v * c11[0] + (1 - u) * v * c01[0],
      (1 - u) * (1 - v) * c00[1] + u * (1 - v) * c10[1] + u * v * c11[1] + (1 - u) * v * c01[1],
    ];
    const ux = (c10[0] - c00[0] + c11[0] - c01[0]) / 2, uz = (c10[1] - c00[1] + c11[1] - c01[1]) / 2;
    const vx = (c01[0] - c00[0] + c11[0] - c10[0]) / 2, vz = (c01[1] - c00[1] + c11[1] - c10[1]) / 2;
    const lenU = Math.hypot(ux, uz), lenV = Math.hypot(vx, vz);
    const ang = Math.atan2(uz, ux);
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const [mx, mz] = bil(0.5, 0.5);
    const old = Math.hypot(mx - OLD_C[0], mz - OLD_C[1]) < 950;
    const r = rand();
    let nx, ny;
    if (old) { nx = r < 0.5 ? 3 : 4; ny = 3; }
    else if (r < 0.13) { nx = 1; ny = 1; }
    else if (r < 0.33) { nx = 2; ny = 1; }
    else if (r < 0.5) { nx = 1; ny = 2; }
    else if (r < 0.86) { nx = 2; ny = 2; }
    else { nx = 3; ny = 2; }
    const m = 8 / s;
    const tall = peakAt(mx, mz) * (1 - smooth(0.6, 1.0, f));
    for (let a = 0; a < nx; a++) for (let b = 0; b < ny; b++) {
      if (rand() < 0.05) continue;
      const u0 = m + (1 - 2 * m) * a / nx, u1 = m + (1 - 2 * m) * (a + 1) / nx;
      const v0 = m + (1 - 2 * m) * b / ny, v1 = m + (1 - 2 * m) * (b + 1) / ny;
      const [x, z] = bil((u0 + u1) / 2, (v0 + v1) / 2);
      let w = (u1 - u0) * lenU * (0.58 + rand() * 0.34);
      let d = (v1 - v0) * lenV * (0.58 + rand() * 0.34);
      let h = 6 + rand() * (old ? 12 : 9);
      h += 104 * tall * Math.pow(rand(), 1.7);
      if (tall > 0.3 && rand() < 0.11 * tall) { h = 62 + rand() * 58 * Math.min(1, tall + 0.2); w *= 0.78; d *= 0.78; }
      h = Math.min(120, Math.max(6, h));
      let ok = true;
      for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]]) {
        const px = x + ca * sx * w / 2 - sa * sz * d / 2, pz = z + sa * sx * w / 2 + ca * sz * d / 2;
        if (onArt(px, pz) || inRiver(px, pz, 22)) { ok = false; break; }
      }
      if (!ok) continue;
      B.x.push(x); B.z.push(z); B.w.push(w); B.d.push(d); B.h.push(h); B.rot.push(ang); B.dist.push(di);
    }
  };

  for (const D of DISTRICTS) {
    const di = D.i;
    const s = D.east ? 98 + rand() * 8 : 110 + rand() * 14;
    const rot = (rand() - 0.5) * 0.55;
    const cs = Math.cos(rot), sn = Math.sin(rot);
    const cache = new Map();
    const corner = (i, j) => {
      const key = (i + 2048) * 4096 + (j + 2048);
      let c = cache.get(key);
      if (!c) {
        const lx = (i + (hash3(i, j, di) - 0.5) * 0.22) * s;
        const lz = (j + (hash3(j, i, di + 17) - 0.5) * 0.22) * s;
        c = [D.x + lx * cs - lz * sn, D.z + lx * sn + lz * cs];
        cache.set(key, c);
      }
      return c;
    };
    const edges = new Set();
    const span = Math.ceil(5000 / s);
    for (let i = -span; i < span; i++) for (let j = -span; j < span; j++) {
      const lx = (i + 0.5) * s, lz = (j + 0.5) * s;
      const x = D.x + lx * cs - lz * sn, z = D.z + lx * sn + lz * cs;
      if (Math.abs(x) > 4700 || Math.abs(z) > 4700) continue;
      const f = cityF(x, z);
      if (f > 1 || districtAt(x, z) !== di) continue;
      const c00 = corner(i, j), c10 = corner(i + 1, j), c11 = corner(i + 1, j + 1), c01 = corner(i, j + 1);
      if (inRiver(c00[0], c00[1], 18) || inRiver(c10[0], c10[1], 18) || inRiver(c11[0], c11[1], 18) || inRiver(c01[0], c01[1], 18)) continue;
      const k0 = ((i + 2048) * 4096 + (j + 2048)) * 2;
      edges.add(k0); edges.add(k0 + 1);
      edges.add(((i + 2048) * 4096 + (j + 2049)) * 2);
      edges.add(((i + 2049) * 4096 + (j + 2048)) * 2 + 1);
      corners.push(c00[0], c00[1]);
      if (rand() < 0.045 + 0.3 * smooth(0.82, 1.0, f)) continue;
      fillBlock(c00, c10, c11, c01, s, f, di);
    }
    for (const key of edges) {
      const dir = key % 2, ij = (key - dir) / 2;
      const i = Math.floor(ij / 4096) - 2048, j = (ij % 4096) - 2048;
      const p = corner(i, j), q = dir === 0 ? corner(i + 1, j) : corner(i, j + 1);
      roadSeg.push(p[0], p[1], q[0], q[1]);
    }
  }

  const bridges = [];
  for (const z of [-2700, -2050, -1250, -400, 350, 950, 1600, 2350]) {
    const x = riverX(z), hw = riverHW(z);
    bridges.push(x - hw - 30, z, x + hw + 30, z + 8);
  }

  const n = B.x.length;
  const buildings = {
    count: n,
    x: Float32Array.from(B.x), z: Float32Array.from(B.z), w: Float32Array.from(B.w), d: Float32Array.from(B.d),
    h: Float32Array.from(B.h), rot: Float32Array.from(B.rot), dist: Uint8Array.from(B.dist),
  };
  return { buildings, roads: Float32Array.from(roadSeg), arterials: Float32Array.from(artSeg), bridges: Float32Array.from(bridges), corners: Float32Array.from(corners) };
}

function placeResidents(city, rand, N) {
  const B = city.buildings;
  const cum = new Float64Array(B.count);
  let acc = 0;
  for (let b = 0; b < B.count; b++) { acc += B.w[b] * B.d[b] * Math.pow(B.h[b], 0.85); cum[b] = acc; }
  const R = {
    count: N, x: new Float32Array(N), y: new Float32Array(N), z: new Float32Array(N),
    b: new Int32Array(N), dist: new Uint8Array(N), band: new Uint8Array(N), age: new Uint8Array(N),
    aff: new Float32Array(N), adW: new Float32Array(N), seed: new Float32Array(N), east: new Uint8Array(N),
  };
  for (let i = 0; i < N; i++) {
    const r = rand() * acc;
    let lo = 0, hi = B.count - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < r) lo = mid + 1; else hi = mid; }
    const b = lo, w = B.w[b], d = B.d[b], h = B.h[b];
    const floors = Math.max(1, Math.floor((h - 2.5) / 3.3));
    const y = Math.min(h - 1.2, 2.4 + Math.floor(rand() * floors) * 3.3);
    const side = rand() < 0.5 ? -1 : 1;
    let lx, lz;
    if (rand() < d / (w + d)) { lx = side * (w / 2 + 0.8); lz = (rand() - 0.5) * 0.84 * d; }
    else { lx = (rand() - 0.5) * 0.84 * w; lz = side * (d / 2 + 0.8); }
    const ca = Math.cos(B.rot[b]), sa = Math.sin(B.rot[b]);
    R.x[i] = B.x[b] + lx * ca - lz * sa;
    R.z[i] = B.z[b] + lx * sa + lz * ca;
    R.y[i] = y;
    R.b[i] = b;
    const di = B.dist[b];
    R.dist[i] = di;
    const east = DISTRICTS[di].east;
    R.east[i] = east ? 1 : 0;
    const probs = east ? AGE_P_EAST : AGE_P_WEST;
    let pr = rand(), band = 0;
    while (band < 4 && pr > probs[band]) { pr -= probs[band]; band++; }
    R.band[i] = band;
    const [a0, a1] = AGE_BANDS[band];
    R.age[i] = a0 + Math.floor(rand() * (a1 - a0 + 1));
    const creator = CREATOR_DISTRICTS.includes(DISTRICTS[di].name);
    R.aff[i] = AGE_AFF[band] * (east ? 0.8 : 1.08) * (creator ? 1.22 : 1) * (0.45 + rand() * 1.1);
    R.adW[i] = AGE_AD[band] * (east ? 0.72 : 1.0);
    R.seed[i] = rand();
  }
  return R;
}

function placeStores(city, R, rand) {
  const SAMPLE = 9000;
  const idx = Array.from({ length: SAMPLE }, () => Math.floor(rand() * R.count));
  const cx = new Float64Array(STORE_N), cz = new Float64Array(STORE_N);
  for (let k = 0; k < STORE_N; k++) { const i = idx[Math.floor(k * SAMPLE / STORE_N)]; cx[k] = R.x[i]; cz[k] = R.z[i]; }
  for (let it = 0; it < 14; it++) {
    const sx = new Float64Array(STORE_N), sz = new Float64Array(STORE_N), sn = new Float64Array(STORE_N);
    for (const i of idx) {
      let best = 0, bd = Infinity;
      for (let k = 0; k < STORE_N; k++) { const dd = (R.x[i] - cx[k]) ** 2 + (R.z[i] - cz[k]) ** 2; if (dd < bd) { bd = dd; best = k; } }
      sx[best] += R.x[i]; sz[best] += R.z[i]; sn[best]++;
    }
    for (let k = 0; k < STORE_N; k++) if (sn[k]) { cx[k] = sx[k] / sn[k]; cz[k] = sz[k] / sn[k]; }
  }
  const C = city.corners;
  const stores = [];
  for (let k = 0; k < STORE_N; k++) {
    let best = 0, bd = Infinity;
    for (let c = 0; c < C.length; c += 2) { const dd = (C[c] - cx[k]) ** 2 + (C[c + 1] - cz[k]) ** 2; if (dd < bd) { bd = dd; best = c; } }
    const x = C[best], z = C[best + 1];
    stores.push({ x, z, district: districtAt(x, z) });
  }
  stores.sort((a, b) => a.x - b.x);
  stores.forEach((s, k) => { s.id = k; s.label = 'Store ' + String(k + 1).padStart(2, '0'); s.name = s.label + ', ' + DISTRICTS[s.district].name; });
  const near1 = new Uint8Array(R.count), near2 = new Uint8Array(R.count), d2 = new Float32Array(R.count);
  const catchment = new Int32Array(STORE_N), inRange = new Int32Array(STORE_N);
  for (let i = 0; i < R.count; i++) {
    let b1 = 0, b2 = 1, e1 = Infinity, e2 = Infinity;
    for (let k = 0; k < STORE_N; k++) {
      const dd = Math.hypot(R.x[i] - stores[k].x, R.z[i] - stores[k].z);
      if (dd < DELIVERY_R) inRange[k]++;
      if (dd < e1) { b2 = b1; e2 = e1; b1 = k; e1 = dd; } else if (dd < e2) { b2 = k; e2 = dd; }
    }
    near1[i] = b1; near2[i] = b2; d2[i] = e2; catchment[b1]++;
  }
  stores.forEach((s, k) => { s.catchment = catchment[k]; s.inRange = inRange[k]; });
  return { stores, near1, near2, d2 };
}

function buildTies(R, rand) {
  const N = R.count, CS = 300, GO = 5000, GN = Math.ceil(2 * GO / CS);
  const cell = new Int32Array(N), counts = new Int32Array(GN * GN + 1);
  for (let i = 0; i < N; i++) {
    const c = Math.floor((R.z[i] + GO) / CS) * GN + Math.floor((R.x[i] + GO) / CS);
    cell[i] = c; counts[c + 1]++;
  }
  for (let c = 0; c < GN * GN; c++) counts[c + 1] += counts[c];
  const fillPos = counts.slice(), members = new Int32Array(N);
  for (let i = 0; i < N; i++) members[fillPos[cell[i]]++] = i;
  const tieStart = new Int32Array(N + 1), list = [];
  for (let i = 0; i < N; i++) {
    tieStart[i] = list.length;
    const k = 3 + Math.floor(rand() * 4);
    const gx = cell[i] % GN, gz = (cell[i] - gx) / GN;
    for (let t = 0; t < k; t++) {
      if (rand() < 0.1) { let j = Math.floor(rand() * N); if (j === i) j = (j + 1) % N; list.push(j); continue; }
      for (let tries = 0; tries < 8; tries++) {
        const ox = Math.round((rand() + rand() - 1) * 2.2), oz = Math.round((rand() + rand() - 1) * 2.2);
        const u = gx + ox, v = gz + oz;
        if (u < 0 || v < 0 || u >= GN || v >= GN) continue;
        const c = v * GN + u, n = counts[c + 1] - counts[c];
        if (!n) continue;
        const j = members[counts[c] + Math.floor(rand() * n)];
        if (j === i || (R.x[j] - R.x[i]) ** 2 + (R.z[j] - R.z[i]) ** 2 > 360000) continue;
        list.push(j); break;
      }
    }
  }
  tieStart[N] = list.length;
  return { tieStart, ties: Int32Array.from(list) };
}

function simulate(R, S, T, rand) {
  const N = R.count, SC = S.stores.length, H = HOURS;
  const { near1, near2, d2 } = S, { ties, tieStart } = T;
  const awareT = new Float32Array(N).fill(1e9), adoptT = new Float32Array(N).fill(1e9);
  const awareSrc = new Int32Array(N).fill(-9), adoptStore = new Int8Array(N).fill(-1);
  const waitUntil = new Int16Array(N).fill(-1), cooldown = new Int16Array(N).fill(-1);
  const soH = new Float32Array(N).fill(-1), soStore = new Int8Array(N).fill(-1), soOutcome = new Uint8Array(N), soAlt = new Int8Array(N).fill(-1);
  const affS = R.aff.map(Math.sqrt);
  const awareList = new Int32Array(N);
  let awareN = 0, adoptN = 0;
  const stock = new Float32Array(SC), planned = new Float32Array(SC), base = new Float32Array(SC), demand = new Float32Array(SC);
  const cap = new Float32Array(SC);
  S.stores.forEach((s, k) => {
    stock[k] = Math.round(s.catchment * P.stockInit); planned[k] = Math.round(s.catchment * P.stockPlan);
    base[k] = s.catchment * P.stockBase; cap[k] = s.catchment * P.capK;
  });
  const stockH = new Float32Array(SC * H), servedH = new Uint16Array(SC * H);
  const awareCount = new Int32Array(H), adoptCount = new Int32Array(H), ordersH = new Int32Array(H), outH = new Uint8Array(H);
  const CAP = 120000;
  const dRes = new Int32Array(CAP), dStore = new Uint8Array(CAP), dT0 = new Float32Array(CAP), dT1 = new Float32Array(CAP);
  let dN = 0;
  const wFrom = new Int32Array(CAP), wTo = new Int32Array(CAP), wT = new Float32Array(CAP);
  let wN = 0;
  const DEC48 = new Float32Array(H + 1).map((_, k) => Math.exp(-k / 48));
  const DEC96 = new Float32Array(H + 1).map((_, k) => Math.exp(-k / 96));
  const TOD_MEAN = TOD.reduce((a, b) => a + b, 0) / 24;
  const adShape = TOD.slice(AD_HOUR, 24); const adSum = adShape.reduce((a, b) => a + b, 0);
  const crShape = Array.from({ length: 30 }, (_, k) => Math.exp(-k / 6) * (0.4 + TALK[(CREATOR_HOUR + k) % 24]));
  const crSum = crShape.reduce((a, b) => a + b, 0);
  const creatorIdx = new Set(CREATOR_DISTRICTS.map(n => DISTRICTS.find(d => d.name === n).i));
  const creatorPool = [];
  for (let i = 0; i < N; i++) if (creatorIdx.has(R.dist[i])) creatorPool.push(i);

  const makeAware = (i, t, src) => {
    awareT[i] = t; awareSrc[i] = src; awareList[awareN++] = i;
    if (src >= 0 && wN < CAP) { wFrom[wN] = src; wTo[wN] = i; wT[wN] = t; wN++; }
  };
  const expose = (count, pool, src, h) => {
    let n = Math.floor(count) + (rand() < count % 1 ? 1 : 0);
    let attempts = n * 10;
    while (n > 0 && attempts-- > 0) {
      const i = pool ? pool[Math.floor(rand() * pool.length)] : Math.floor(rand() * N);
      if (awareT[i] < 1e9) { n--; continue; }
      if (rand() < R.adW[i]) { makeAware(i, h + rand(), src); n--; }
    }
  };
  const placeOrder = (i, s, first, h, rain) => {
    stock[s] -= 1; servedH[s * H + h]++; ordersH[h]++;
    const t0 = Math.max(h + rand() * 0.97, awareT[i] + 0.04);
    const dur = (0.15 + rand() * 0.2) * (rain ? 1.35 : 1);
    if (dN < CAP) { dRes[dN] = i; dStore[dN] = s; dT0[dN] = t0; dT1[dN] = t0 + dur; dN++; }
    if (first) { adoptT[i] = t0; adoptStore[i] = s; adoptN++; }
  };
  const tryOrder = (i, first, h, rain) => {
    const s1 = near1[i];
    demand[s1]++;
    if (stock[s1] >= 1) { placeOrder(i, s1, first, h, rain); return; }
    const roll = rand(), s2 = near2[i];
    const spill = d2[i] < SPILL_R && stock[s2] >= 1;
    let outcome;
    if (roll < 0.42 && spill) { outcome = 1; placeOrder(i, s2, first, h, rain); }
    else if (roll < 0.8) { outcome = 2; waitUntil[i] = h + 24; }
    else { outcome = 3; if (first) cooldown[i] = h + 96; }
    if (soH[i] < 0) { soH[i] = h + rand() * 0.9; soStore[i] = s1; soOutcome[i] = outcome; soAlt[i] = outcome === 1 ? s2 : -1; }
  };

  for (let h = 0; h < H; h++) {
    const hod = h % 24, tod = TOD[hod], talk = TALK[hod];
    const rain = h >= RAIN_A && h < RAIN_B;
    const eve = rain && hod >= 17 ? 1.12 : 1;
    for (let s = 0; s < SC; s++) {
      if (h % 48 !== RESTOCK_HOUR + (s % 2) * 24) continue;
      {
        const launch = h < P.capUntil;
        stock[s] += launch ? planned[s] : Math.max(planned[s], Math.round(base[s] + P.safety * demand[s]));
        planned[s] = Math.round(launch ? Math.min(cap[s], base[s] + P.restockK * demand[s]) : base[s] + P.safety * demand[s]);
        demand[s] = 0;
      }
    }
    if (h >= AD_HOUR && h < 24) expose(N * P.adDay1 * adShape[h - AD_HOUR] / adSum, null, -1, h);
    else if (h >= 24) expose(N * P.adDrip / 24 * tod / TOD_MEAN, null, -3, h);
    if (h >= CREATOR_HOUR && h < CREATOR_HOUR + 30) {
      const k = crShape[h - CREATOR_HOUR] / crSum;
      expose(creatorPool.length * P.creatorIn * k, creatorPool, -2, h);
      expose(N * P.creatorOut * k, null, -2, h);
    }
    const n0 = awareN;
    for (let q = 0; q < n0; q++) {
      const i = awareList[q];
      const adopted = adoptT[i] < 1e9;
      const since = Math.min(H, h - Math.floor(awareT[i]));
      const pt = adopted ? P.betaA * (h - adoptT[i] < 72 ? 1 : 0.35) : P.betaW * DEC48[since];
      if (rand() < pt * talk) {
        const a = tieStart[i], n = tieStart[i + 1] - a;
        if (n > 0) {
          const j = ties[a + Math.floor(rand() * n)];
          if (awareT[j] >= 1e9) makeAware(j, Math.max(h + rand(), awareT[i] + 0.1), i);
        }
      }
      if (waitUntil[i] === h) { waitUntil[i] = -1; tryOrder(i, !adopted, h, rain); continue; }
      if (adopted) {
        if (rand() < P.r0 * tod * affS[i] * eve) tryOrder(i, false, h, rain);
      } else if (cooldown[i] <= h && waitUntil[i] < h) {
        if (rand() < P.a0 * R.aff[i] * tod * eve * (0.15 + 0.85 * DEC96[since])) tryOrder(i, true, h, rain);
      }
    }
    let out = 0;
    for (let s = 0; s < SC; s++) {
      stockH[s * H + h] = Math.max(0, stock[s]);
      if (stock[s] < 1) out++;
    }
    outH[h] = out; awareCount[h] = awareN; adoptCount[h] = adoptN;
  }

  const outIntervals = S.stores.map((_, s) => {
    const list = [];
    let start = -1;
    for (let h = 0; h <= H; h++) {
      const isOut = h < H && stockH[s * H + h] < 1;
      if (isOut && start < 0) start = h;
      if (!isOut && start >= 0) { list.push([start, h]); start = -1; }
    }
    return list;
  });

  return {
    awareT, adoptT, awareSrc, adoptStore, soH, soStore, soOutcome, soAlt,
    stockH, servedH, awareCount, adoptCount, ordersH, outH, outIntervals,
    deliveries: { n: dN, res: dRes.subarray(0, dN), store: dStore.subarray(0, dN), t0: dT0.subarray(0, dN), t1: dT1.subarray(0, dN) },
    wom: { n: wN, from: wFrom.subarray(0, wN), to: wTo.subarray(0, wN), t: wT.subarray(0, wN) },
  };
}

function buildWorld(N) {
  const rand = mulberry32(49);
  const t0 = globalThis.performance.now();
  const city = buildCity(rand);
  const res = placeResidents(city, rand, N);
  const S = placeStores(city, res, rand);
  const T = buildTies(res, rand);
  const t1 = globalThis.performance.now();
  const sim = simulate(res, S, T, rand);
  const t2 = globalThis.performance.now();
  return { city, res, stores: S.stores, near1: S.near1, ties: T, sim, timing: { city: t1 - t0, sim: t2 - t1 } };
}

function summary(world) {
  const { sim, city, stores, res } = world;
  const day = (d) => Math.min(HOURS, d * 24) - 1;
  const pct = (n) => Math.round((1000 * n) / res.count) / 10;
  const rows = [1, 6, 10, 13, 20, 30].map((d) => ({
    day: d, heard: pct(sim.awareCount[day(d)]), ordered: pct(sim.adoptCount[day(d)]), storesOut: sim.outH[day(d)],
  }));
  return {
    residents: res.count, buildings: city.buildings.count, stores: stores.length,
    ties: world.ties.ties.length, deliveries: sim.deliveries.n, chatter: sim.wom.n,
    days: rows, timingMs: { city: Math.round(world.timing.city), sim: Math.round(world.timing.sim) },
  };
}

export { HOURS, P, DISTRICTS, mulberry32, buildCity, placeResidents, placeStores, buildTies, simulate, buildWorld, summary };

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  const n = Number(process.argv[2] || 48000);
  console.log(JSON.stringify(summary(buildWorld(n)), null, 2));
}
