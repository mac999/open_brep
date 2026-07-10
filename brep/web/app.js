/*
 * B-Rep Authoring Tool - front end.
 *
 * The server tessellates every kernel face into triangles, so this file draws
 * them straight onto a 2D canvas (projection + painter's algorithm) with no
 * external 3D library. Click picking reuses the same screen coordinates.
 */
'use strict';

const PALETTE = [
  [90, 148, 214], [214, 141, 90], [122, 190, 122], [190, 122, 190],
  [204, 178, 92], [110, 186, 196], [200, 110, 120], [150, 150, 180],
];
const SELECTED = [77, 163, 255];

/* One colour per pickable entity type, shared by the mode buttons (CSS),
 * the canvas grips, the selection highlight and the status line — the colour
 * IS the mode, so the UI reads consistently. */
const TYPE_COLORS = { solid: '#4da3ff', face: '#ffc454', edge: '#5ad1e6', vertex: '#7ee787' };
const FACE_SEL_RGB = [255, 196, 84];
const FACE_HOVER_BLEND = 0.35;   // how far a hovered face tints toward white

const PRIMS = {
  box:      { label: 'Box',      params: [['length', 'Length L', 10], ['width', 'Width W', 10], ['height', 'Height H', 10]] },
  sphere:   { label: 'Sphere',   params: [['radius', 'Radius', 5], ['slices', 'Slices', 16], ['stacks', 'Stacks', 8]] },
  cylinder: { label: 'Cylinder', params: [['radius', 'Radius', 4], ['height', 'Height', 10], ['slices', 'Slices', 16]] },
  plane:    { label: 'Plane',    params: [['width', 'Width', 20], ['height', 'Height', 20]] },
  nurbs:    { label: 'NURBS Dome', params: [['size', 'Size', 20], ['height', 'Height', 8]] },
};

/*
 * Modeling operations exposed as menus. Each entry declares its input fields
 * and a builder that assembles the exact CLI command line; the line is then
 * sent through POST /api/command, i.e. the same shell.onecmd() dispatch the
 * REPL uses. Parsing, validation and messages are all reused from the CLI.
 *
 * Field types:
 *   num              numeric input
 *   select           fixed choice list (options)
 *   text             free text (used for optional clauses)
 *   solid|face|edge  id picker filled from GET /api/entities
 *   entity           id picker offering both faces and edges
 */
const OPS = {
  extrude: {
    label: 'Extrude face → prism',
    hint: 'Sweep a planar face along a vector. CLI: extrude #<face> <dx> <dy> <dz>',
    fields: [
      { key: 'face', type: 'face', label: 'Face' },
      { key: 'dx', type: 'num', label: 'dx', def: 0 },
      { key: 'dy', type: 'num', label: 'dy', def: 0 },
      { key: 'dz', type: 'num', label: 'dz', def: 10 },
    ],
    build: (v) => `extrude #${v.face} ${v.dx} ${v.dy} ${v.dz}`,
  },
  revolve: {
    label: 'Revolve face → solid',
    hint: 'Faceted rotational sweep about a principal axis. CLI: revolve #<face> <x|y|z> <angle> [segments]',
    fields: [
      { key: 'face', type: 'face', label: 'Face' },
      { key: 'axis', type: 'select', label: 'Axis', options: ['x', 'y', 'z'], def: 'z' },
      { key: 'angle', type: 'num', label: 'Angle °', def: 360 },
      { key: 'segments', type: 'num', label: 'Segments', def: 8 },
    ],
    build: (v) => `revolve #${v.face} ${v.axis} ${v.angle} ${v.segments}`,
  },
  trimPlane: {
    label: 'Trim solid by plane',
    hint: 'Half-space cut by the plane nx·x + ny·y + nz·z = d. CLI: trim #<solid> by plane <nx> <ny> <nz> <d> keep <side>',
    fields: [
      { key: 'solid', type: 'solid', label: 'Solid' },
      { key: 'nx', type: 'num', label: 'nx', def: 0 },
      { key: 'ny', type: 'num', label: 'ny', def: 0 },
      { key: 'nz', type: 'num', label: 'nz', def: 1 },
      { key: 'd', type: 'num', label: 'd', def: 0 },
      { key: 'keep', type: 'select', label: 'Keep side', options: ['above', 'below'], def: 'above' },
    ],
    build: (v) => `trim #${v.solid} by plane ${v.nx} ${v.ny} ${v.nz} ${v.d} keep ${v.keep}`,
  },
  trimSurface: {
    label: 'Trim solid by NURBS face',
    hint: 'Half-space cut by a curved cutter surface. CLI: trim #<solid> by surface #<face> keep <side>',
    fields: [
      { key: 'solid', type: 'solid', label: 'Solid' },
      { key: 'cutter', type: 'face', label: 'Cutter face (NURBS)' },
      { key: 'keep', type: 'select', label: 'Keep side', options: ['above', 'below'], def: 'above' },
    ],
    build: (v) => `trim #${v.solid} by surface #${v.cutter} keep ${v.keep}`,
  },
  trimWindow: {
    label: 'Trim surface (u,v window)',
    hint: 'Crop a NURBS face to the kept parametric window. CLI: trim surface #<face> keep <u0> <u1> <v0> <v1>',
    fields: [
      { key: 'face', type: 'face', label: 'Face (NURBS)' },
      { key: 'u0', type: 'num', label: 'u0', def: 0.25 },
      { key: 'u1', type: 'num', label: 'u1', def: 0.75 },
      { key: 'v0', type: 'num', label: 'v0', def: 0.25 },
      { key: 'v1', type: 'num', label: 'v1', def: 0.75 },
    ],
    build: (v) => `trim surface #${v.face} keep ${v.u0} ${v.u1} ${v.v0} ${v.v1}`,
  },
  trimCurve: {
    label: 'Split edge at parameter',
    hint: 'Split an edge at u ∈ (0,1); both halves are kept. CLI: trim curve #<edge> at <u>',
    fields: [
      { key: 'edge', type: 'edge', label: 'Edge' },
      { key: 'u', type: 'num', label: 'u (0..1)', def: 0.5 },
    ],
    build: (v) => `trim curve #${v.edge} at ${v.u}`,
  },
  extendPlane: {
    label: 'Extend to plane',
    hint: 'Grow an edge or face until it reaches the plane. CLI: extend #<edge|face> to plane <nx> <ny> <nz> <d> [along <dx dy dz>]',
    fields: [
      { key: 'src', type: 'entity', label: 'Edge / Face' },
      { key: 'nx', type: 'num', label: 'nx', def: 0 },
      { key: 'ny', type: 'num', label: 'ny', def: 0 },
      { key: 'nz', type: 'num', label: 'nz', def: 1 },
      { key: 'd', type: 'num', label: 'd', def: 10 },
      { key: 'along', type: 'text', label: 'along dx dy dz (optional, faces only)', def: '' },
    ],
    build: (v) => `extend #${v.src} to plane ${v.nx} ${v.ny} ${v.nz} ${v.d}${alongSuffix(v.along)}`,
  },
  extendFace: {
    label: 'Extend to face',
    hint: 'Grow an edge or face onto a target face (its plane, or its NURBS surface). CLI: extend #<edge|face> to #<face> [along <dx dy dz>]',
    fields: [
      { key: 'src', type: 'entity', label: 'Edge / Face' },
      { key: 'target', type: 'face', label: 'Target face' },
      { key: 'along', type: 'text', label: 'along dx dy dz (optional, faces only)', def: '' },
    ],
    build: (v) => `extend #${v.src} to #${v.target}${alongSuffix(v.along)}`,
  },
  intersect: {
    label: 'Intersect NURBS × NURBS',
    hint: 'Surface-surface intersection curve as a wire solid. CLI: intersect #<faceA> #<faceB> samples <n>',
    fields: [
      { key: 'a', type: 'face', label: 'Face A (NURBS)' },
      { key: 'b', type: 'face', label: 'Face B (NURBS)' },
      { key: 'samples', type: 'num', label: 'Samples', def: 32 },
    ],
    build: (v) => `intersect #${v.a} #${v.b} samples ${v.samples}`,
  },
  blend: {
    label: 'Blend (G2 patch)',
    hint: 'Curvature-continuous blend strip across the intersection. CLI: blend #<faceA> #<faceB> width <w> samples <n>',
    fields: [
      { key: 'a', type: 'face', label: 'Face A (NURBS)' },
      { key: 'b', type: 'face', label: 'Face B (NURBS)' },
      { key: 'width', type: 'num', label: 'Width', def: 1.5 },
      { key: 'samples', type: 'num', label: 'Samples', def: 9 },
    ],
    build: (v) => `blend #${v.a} #${v.b} width ${v.width} samples ${v.samples}`,
  },
};

/* The left panel splits the ops by conceptual weight: sweeps act on one face
 * of one shape; combine ops relate two shapes (or cut one against a surface). */
const OP_GROUPS = {
  sweep: ['extrude', 'revolve'],
  combine: ['trimPlane', 'trimSurface', 'trimWindow', 'trimCurve',
            'extendPlane', 'extendFace', 'intersect', 'blend'],
};

function alongSuffix(text) {
  const t = (text || '').trim();
  if (!t) return '';
  const nums = t.split(/[\s,]+/).map(Number);
  if (nums.length !== 3 || nums.some((n) => !Number.isFinite(n))) {
    throw new Error("'along' must be three numbers: dx dy dz");
  }
  return ` along ${nums.join(' ')}`;
}

const state = {
  solids: [],
  entities: [],       // per-solid surviving face/edge/vertex ids (op pickers)
  selected: null,     // selected solid oid
  selMode: 'solid',   // canvas picking mode: solid | face | edge | vertex
  subSel: null,       // picked sub-entity: {type, oid, solid} (face/edge/vertex)
  subInfo: null,      // /api/entity detail for subSel (drives the props panel)
  shade: 'solid',
  prim: 'box',
  op: { sweep: 'extrude', combine: 'trimPlane' },
  cam: { yaw: -0.9, pitch: 0.55, dist: 90, target: [0, 0, 0] },
  history: [],
  historyIndex: 0,
};

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
let hitBuffer = [];   // screen triangles of the last render (near-to-far), for picking
let edgeHits = [];    // screen segments of the last render (edge picking)
let vertHits = [];    // screen points of the last render (vertex picking)
let faceGripHits = []; // face grip handles of the last render (face picking)
let hover = null;     // entity under the cursor: {type, oid, solid, px, py}
let camDragging = false;  // true while orbiting/panning; grips hide meanwhile

/* ── vector utils ──────────────────────────────────────────────── */
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/* ── server API ────────────────────────────────────────────────── */
async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ ok: false, error: 'could not parse the server response' }));
  if (!data.ok) throw new Error(data.error || `request failed (${response.status})`);
  return data;
}

async function refreshScene() {
  const [scene, entities] = await Promise.all([api('/api/scene'), api('/api/entities')]);
  state.solids = scene.solids;
  state.entities = entities.solids;
  document.getElementById('cwd').textContent = scene.cwd;
  if (state.selected !== null && !state.solids.some((s) => s.oid === state.selected)) {
    state.selected = null;
  }
  await syncSubSel();
  hover = null;   // the entity under the cursor may not exist any more
  renderSolidList();
  renderProps();
  renderOpParams('sweep');
  renderOpParams('combine');
  updateSelStatus();
  draw();
}

/* A trim/extend may consume the picked face or edge: keep the sub-selection
 * only while its id still survives, and refresh its properties otherwise. */
async function syncSubSel() {
  if (!state.subSel) { state.subInfo = null; return; }
  const { type, oid, solid } = state.subSel;
  const owner = state.entities.find((s) => s.oid === solid);
  const alive = owner && (
    (type === 'face' && owner.faces.some((f) => f.oid === oid)) ||
    (type === 'edge' && owner.edges.includes(oid)) ||
    (type === 'vertex' && owner.vertices.includes(oid)));
  if (!alive) { state.subSel = null; state.subInfo = null; return; }
  try {
    state.subInfo = (await api(`/api/entity?oid=${oid}`)).entity;
  } catch (err) {
    state.subSel = null;
    state.subInfo = null;
  }
}

/* Run one CLI line through the shared shell (POST /api/command) and echo the
 * exchange in the console pane — the same path the console input uses. */
async function runCommand(line) {
  log(`brep> ${line}`, 'cmd');
  const data = await api('/api/command', { line });
  // The shell prints CLI failures as 'ERROR: ...' lines; colour them like errors.
  if (data.output) log(data.output, /^ERROR:/m.test(data.output) ? 'err' : '');
  await refreshScene();
}

/* ── camera ────────────────────────────────────────────────────── */
function cameraBasis() {
  const { yaw, pitch, dist, target } = state.cam;
  const offset = [
    dist * Math.cos(pitch) * Math.cos(yaw),
    dist * Math.cos(pitch) * Math.sin(yaw),
    dist * Math.sin(pitch),
  ];
  const eye = [target[0] + offset[0], target[1] + offset[1], target[2] + offset[2]];
  const forward = norm(sub(target, eye));
  let right = cross(forward, [0, 0, 1]);
  if (Math.hypot(right[0], right[1], right[2]) < 1e-6) right = [1, 0, 0];
  right = norm(right);
  const up = cross(right, forward);
  return { eye, forward, right, up };
}

function fitView() {
  const points = state.solids.flatMap((s) => [s.bbox.min, s.bbox.max]);
  if (!points.length) { state.cam.target = [0, 0, 0]; state.cam.dist = 90; draw(); return; }
  const lo = [0, 1, 2].map((i) => Math.min(...points.map((p) => p[i])));
  const hi = [0, 1, 2].map((i) => Math.max(...points.map((p) => p[i])));
  state.cam.target = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
  const radius = Math.max(1e-3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
  state.cam.dist = radius * 3.2;
  draw();
}

/* ── rendering ─────────────────────────────────────────────────── */
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = canvas;
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#101318';
  ctx.fillRect(0, 0, w, h);

  const { eye, forward, right, up } = cameraBasis();
  const focal = h / (2 * Math.tan(0.5 * 0.9));   // ~51 deg vertical FOV
  const near = 0.05;

  // world -> screen; null when behind the camera (z <= near).
  const project = (p) => {
    const d = sub(p, eye);
    const z = dot(d, forward);
    if (z <= near) return null;
    return [w / 2 + (dot(d, right) * focal) / z, h / 2 - (dot(d, up) * focal) / z, z];
  };

  drawGrid(project);

  const subSel = state.subSel;
  const faces = [];
  const lines = [];
  const grips = [];   // candidate pick handles; occlusion-culled once hitBuffer is fresh
  edgeHits = [];
  vertHits = [];
  for (const solid of state.solids) {
    const isSel = solid.oid === state.selected;
    const base = isSel ? SELECTED : PALETTE[solid.oid % PALETTE.length];
    const screen = solid.positions.map(project);

    // Triangles are collected even in wire mode: they are the picking surface.
    solid.triangles.forEach(([ia, ib, ic], t) => {
      const sa = screen[ia]; const sb = screen[ib]; const sc = screen[ic];
      if (!sa || !sb || !sc) return;
      const [pa, pb, pc] = [solid.positions[ia], solid.positions[ib], solid.positions[ic]];
      const n = norm(cross(sub(pb, pa), sub(pc, pa)));
      const centre = [(pa[0] + pb[0] + pc[0]) / 3, (pa[1] + pb[1] + pc[1]) / 3, (pa[2] + pb[2] + pc[2]) / 3];
      const toEye = norm(sub(eye, centre));
      const lambert = 0.3 + 0.7 * Math.abs(dot(n, toEye));
      const faceOid = solid.triFaces ? solid.triFaces[t] : null;
      const isFaceSel = subSel && subSel.type === 'face' && subSel.oid === faceOid;
      const isFaceHover = !camDragging && hover && hover.type === 'face' && hover.oid === faceOid;
      let tint = isFaceSel ? FACE_SEL_RGB : base;
      if (isFaceHover && !isFaceSel) {
        tint = tint.map((c) => c + (255 - c) * FACE_HOVER_BLEND);
      }
      faces.push({
        depth: (sa[2] + sb[2] + sc[2]) / 3,
        screen: [sa, sb, sc],
        oid: solid.oid,
        face: faceOid,
        fill: `rgb(${tint.map((c) => Math.round(c * lambert)).join(',')})`,
      });
    });

    solid.wire.forEach(([a, b], i) => {
      const sa = project(a); const sb = project(b);
      if (!sa || !sb) return;
      const edgeOid = solid.wireEdges ? solid.wireEdges[i] : null;
      const isEdgeSel = subSel && subSel.type === 'edge' && subSel.oid === edgeOid;
      const isEdgeHover = !camDragging && hover && hover.type === 'edge' && hover.oid === edgeOid;
      lines.push({ depth: (sa[2] + sb[2]) / 2, a: sa, b: sb, selected: isSel,
                   edgeSel: isEdgeSel, edgeHover: isEdgeHover });
      if (state.selMode === 'edge' && edgeOid !== null) {
        edgeHits.push({ a: sa, b: sb, oid: edgeOid, solid: solid.oid, depth: (sa[2] + sb[2]) / 2 });
        grips.push({ type: 'edge', oid: edgeOid, solid: solid.oid,
                     x: (sa[0] + sb[0]) / 2, y: (sa[1] + sb[1]) / 2,
                     depth: (sa[2] + sb[2]) / 2, selected: isEdgeSel });
      }
    });

    if (state.selMode === 'face') {
      for (const g of solid.faceGrips || []) {
        const sp = project(g.p);
        if (!sp) continue;
        grips.push({ type: 'face', oid: g.oid, solid: solid.oid,
                     x: sp[0], y: sp[1], depth: sp[2],
                     selected: subSel && subSel.type === 'face' && subSel.oid === g.oid });
      }
    }

    if (state.selMode === 'vertex' || (subSel && subSel.type === 'vertex')) {
      for (const v of solid.verts || []) {
        const sp = project(v.p);
        if (!sp) continue;
        const isVertSel = subSel && subSel.type === 'vertex' && subSel.oid === v.oid;
        if (state.selMode === 'vertex') {
          vertHits.push({ x: sp[0], y: sp[1], oid: v.oid, solid: solid.oid, depth: sp[2] });
        }
        if (state.selMode === 'vertex' || isVertSel) {
          grips.push({ type: 'vertex', oid: v.oid, solid: solid.oid,
                       x: sp[0], y: sp[1], depth: sp[2], selected: isVertSel });
        }
      }
    }
  }

  faces.sort((p, q) => q.depth - p.depth);          // far-to-near
  if (state.shade === 'solid') {
    for (const f of faces) {
      ctx.beginPath();
      ctx.moveTo(f.screen[0][0], f.screen[0][1]);
      ctx.lineTo(f.screen[1][0], f.screen[1][1]);
      ctx.lineTo(f.screen[2][0], f.screen[2][1]);
      ctx.closePath();
      ctx.fillStyle = f.fill;
      ctx.fill();
    }
  }

  lines.sort((p, q) => q.depth - p.depth);
  for (const l of lines) {
    ctx.beginPath();
    ctx.moveTo(l.a[0], l.a[1]);
    ctx.lineTo(l.b[0], l.b[1]);
    if (l.edgeSel || l.edgeHover) {
      ctx.strokeStyle = l.edgeSel ? TYPE_COLORS.edge : '#bdeef7';
      ctx.lineWidth = l.edgeSel ? 2.6 : 2;
    } else {
      ctx.strokeStyle = l.selected ? '#cfe6ff' : (state.shade === 'wire' ? '#7f8b9c' : 'rgba(12,14,18,.55)');
      ctx.lineWidth = l.selected ? 1.6 : 1;
    }
    ctx.stroke();
  }

  hitBuffer = faces.slice().reverse();              // nearest first, for picking

  // Grips are culled against the *fresh* hitBuffer (a stale one would flicker
  // after camera moves) and hidden entirely while the camera is being dragged.
  faceGripHits = [];
  if (!camDragging) {
    for (const g of grips) {
      if (!g.selected && occludedAt(g.x, g.y, g.depth)) continue;
      if (g.type === 'face') faceGripHits.push(g);
      drawGrip(g);
    }
    drawHoverLabel();
  }

  const total = state.solids.reduce((n, s) => n + s.stats.f, 0);
  document.getElementById('scene-info').textContent =
    state.solids.length ? `${state.solids.length} solid(s) - ${total} faces` : 'No solids';
}

/* One pick handle. Shape encodes the type (square = vertex, diamond = edge,
 * circle = face); colour matches the mode buttons and the status line. */
function drawGrip(g) {
  const isHover = hover && hover.type === g.type && hover.oid === g.oid;
  const size = (g.selected ? 5 : 3.5) + (isHover ? 1.5 : 0);
  const colour = TYPE_COLORS[g.type];
  ctx.beginPath();
  if (g.type === 'vertex') {
    ctx.rect(g.x - size, g.y - size, size * 2, size * 2);
  } else if (g.type === 'edge') {
    ctx.moveTo(g.x, g.y - size - 1);
    ctx.lineTo(g.x + size + 1, g.y);
    ctx.lineTo(g.x, g.y + size + 1);
    ctx.lineTo(g.x - size - 1, g.y);
    ctx.closePath();
  } else {
    ctx.arc(g.x, g.y, size, 0, Math.PI * 2);
  }
  ctx.fillStyle = g.selected || isHover ? colour : 'rgba(20,23,28,.85)';
  ctx.fill();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.4;
  ctx.stroke();
}

/* Small name tag next to whatever the cursor is over, so a pick target is
 * identified before it is clicked. */
function drawHoverLabel() {
  if (!hover) return;
  const owner = state.solids.find((s) => s.oid === hover.solid);
  const text = hover.type === 'solid'
    ? `${owner ? owner.name : 'solid'} #${hover.oid}`
    : `${hover.type} #${hover.oid}`;
  ctx.font = '11px Consolas, monospace';
  const w = ctx.measureText(text).width + 12;
  const x = Math.min(hover.px + 14, canvas.clientWidth - w - 4);
  const y = Math.max(hover.py - 26, 4);
  ctx.fillStyle = 'rgba(20,23,28,.92)';
  ctx.strokeStyle = TYPE_COLORS[hover.type] || '#4da3ff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, 18, 4);
  else ctx.rect(x, y, w, 18);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#dfe4ec';
  ctx.fillText(text, x + 6, y + 13);
}

function drawGrid(project) {
  const span = Math.max(20, state.cam.dist * 0.6);
  const step = Math.pow(10, Math.round(Math.log10(span / 10)));
  const extent = Math.ceil(span / step) * step;
  ctx.lineWidth = 1;
  for (let t = -extent; t <= extent + 1e-9; t += step) {
    for (const seg of [[[t, -extent, 0], [t, extent, 0]], [[-extent, t, 0], [extent, t, 0]]]) {
      const a = project(seg[0]); const b = project(seg[1]);
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.strokeStyle = Math.abs(t) < 1e-9 ? 'rgba(120,140,170,.35)' : 'rgba(90,105,130,.13)';
      ctx.stroke();
    }
  }
}

/* Nearest triangle under the cursor: {oid, face} or null. */
function pickTriangle(px, py) {
  const inside = (p, [a, b, c]) => {
    const sign = (u, v, x) => (x[0] - v[0]) * (u[1] - v[1]) - (u[0] - v[0]) * (x[1] - v[1]);
    const d1 = sign(p, a, b); const d2 = sign(p, b, c); const d3 = sign(p, c, a);
    const neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(neg && pos);
  };
  for (const f of hitBuffer) if (inside([px, py], f.screen)) return f;
  return null;
}

/* Distance from (px,py) to segment a-b, plus the parameter t of the nearest
 * point — t interpolates the segment's depth for the occlusion test. */
function segNearest(px, py, a, b) {
  const vx = b[0] - a[0]; const vy = b[1] - a[1];
  const len2 = vx * vx + vy * vy;
  const t = len2 ? Math.max(0, Math.min(1, ((px - a[0]) * vx + (py - a[1]) * vy) / len2)) : 0;
  return { d: Math.hypot(px - (a[0] + t * vx), py - (a[1] + t * vy)), t };
}

/* Screen-space barycentric depth of triangle f at (px,py). */
function triDepthAt(px, py, f) {
  const [a, b, c] = f.screen;
  const det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  if (Math.abs(det) < 1e-9) return f.depth;
  const w1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / det;
  const w2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / det;
  return w1 * a[2] + w2 * b[2] + (1 - w1 - w2) * c[2];
}

/* In shaded mode a vertex/edge behind a face is not a valid pick target.
 * 2% depth slack keeps entities lying ON the front faces pickable; very thin
 * solids degrade to see-through picking, which is the safe direction. */
function occludedAt(px, py, depth) {
  if (state.shade !== 'solid') return false;
  const front = pickTriangle(px, py);
  return front !== null && depth > triDepthAt(px, py, front) * 1.02;
}

/* Resolve a click into the entity of the active selection mode. Returns
 * {type, oid, solid} (type 'solid' carries the solid oid) or null. */
function pick(px, py) {
  if (state.selMode === 'vertex') {
    let best = null;
    for (const v of vertHits) {
      const d = Math.hypot(px - v.x, py - v.y);
      if (d > 10 || occludedAt(v.x, v.y, v.depth)) continue;
      if (!best || d < best.d - 0.5 || (Math.abs(d - best.d) <= 0.5 && v.depth < best.depth)) {
        best = { d, depth: v.depth, oid: v.oid, solid: v.solid };
      }
    }
    return best ? { type: 'vertex', oid: best.oid, solid: best.solid } : null;
  }
  if (state.selMode === 'edge') {
    let best = null;
    for (const e of edgeHits) {
      const { d, t } = segNearest(px, py, e.a, e.b);
      if (d > 8) continue;
      const depth = e.a[2] + t * (e.b[2] - e.a[2]);
      const qx = e.a[0] + t * (e.b[0] - e.a[0]);
      const qy = e.a[1] + t * (e.b[1] - e.a[1]);
      if (occludedAt(qx, qy, depth)) continue;
      if (!best || d < best.d - 0.5 || (Math.abs(d - best.d) <= 0.5 && depth < best.depth)) {
        best = { d, depth, oid: e.oid, solid: e.solid };
      }
    }
    return best ? { type: 'edge', oid: best.oid, solid: best.solid } : null;
  }
  if (state.selMode === 'face') {
    // A grip is a precise target for small or edge-on faces; the face body
    // remains clickable as before.
    let bestGrip = null;
    for (const g of faceGripHits) {
      const d = Math.hypot(px - g.x, py - g.y);
      if (d <= 9 && (!bestGrip || d < bestGrip.d)) bestGrip = { d, g };
    }
    if (bestGrip) {
      return { type: 'face', oid: bestGrip.g.oid, solid: bestGrip.g.solid };
    }
  }
  const hit = pickTriangle(px, py);
  if (!hit) return null;
  if (state.selMode === 'face' && hit.face !== null) {
    return { type: 'face', oid: hit.face, solid: hit.oid };
  }
  return { type: 'solid', oid: hit.oid, solid: hit.oid };
}

/* ── right panel ───────────────────────────────────────────────── */
function renderSolidList() {
  const list = document.getElementById('solid-list');
  list.innerHTML = '';
  if (!state.solids.length) {
    list.innerHTML = '<li class="empty">No solids yet.</li>';
    return;
  }
  for (const s of state.solids) {
    const colour = PALETTE[s.oid % PALETTE.length];
    const li = document.createElement('li');
    li.className = s.oid === state.selected ? 'selected' : '';
    li.innerHTML =
      `<span class="swatch" style="background:rgb(${colour.join(',')})"></span>` +
      `<span class="s-name">${escapeHtml(s.kind)}</span>` +
      `<span class="vef" title="vertices / edges / faces">V${s.stats.v} E${s.stats.e} F${s.stats.f}</span>` +
      `<span class="oid">#${s.oid}</span>`;
    // Clicking the selected row again toggles the selection off.
    li.onclick = () => select(s.oid === state.selected ? null : s.oid);
    list.appendChild(li);
  }
}

function renderProps() {
  const box = document.getElementById('props');
  const solid = state.solids.find((s) => s.oid === state.selected);
  const hasSelection = Boolean(solid);
  for (const id of ['btn-move', 'btn-delete', 'btn-scale', 'btn-rot-pos',
                    'btn-rot-neg', 'btn-deselect']) {
    document.getElementById(id).disabled = !hasSelection;
  }
  document.querySelectorAll('[data-nudge]').forEach((b) => { b.disabled = !hasSelection; });

  if (!solid) {
    box.className = 'props empty';
    box.textContent = 'Select a solid (or a face/edge/vertex) to inspect and edit it.';
    return;
  }
  box.className = 'props';
  const f3 = (v) => v.map((n) => n.toFixed(2)).join(', ');
  const f = (n) => Number(n.toFixed(3));
  const size = [0, 1, 2].map((i) => solid.bbox.max[i] - solid.bbox.min[i]);
  const st = solid.stats;
  const badge = solid.valid
    ? '<span class="badge ok">valid</span>'
    : '<span class="badge bad">invalid</span>';
  box.innerHTML = `
    ${subPropsHtml()}
    <h3>Edit</h3>
    <div class="edit-grid">
      <span>Name</span>
      <input id="prop-name" type="text" spellcheck="false" value="${escapeHtml(solid.name)}">
      <span>Center</span>
      <div class="edit-vec">
        <input id="prop-cx" type="number" step="any" value="${f(solid.centroid[0])}">
        <input id="prop-cy" type="number" step="any" value="${f(solid.centroid[1])}">
        <input id="prop-cz" type="number" step="any" value="${f(solid.centroid[2])}">
      </div>
    </div>
    <div class="apply-row">
      <button id="prop-apply" class="primary">Apply Changes</button>
    </div>
    <h3>Identity</h3>
    <dl>
      <dt>ID</dt><dd>#${solid.oid}</dd>
      <dt>Kind</dt><dd>${escapeHtml(solid.kind)}</dd>
    </dl>
    <h3>Topology</h3>
    <dl>
      <dt>V / E / F</dt><dd>${st.v} / ${st.e} / ${st.f}</dd>
      <dt>Rings</dt><dd>${st.rings}</dd>
      <dt>Shells / Genus</dt><dd>${st.shells} / ${st.genus}</dd>
      <dt>NURBS faces</dt><dd>${st.nurbsFaces}</dd>
      <dt>Euler</dt><dd>${solid.eulerLhs} = ${solid.eulerRhs} ${badge}</dd>
    </dl>
    <h3>Entities</h3>
    <div class="ent-browse">
      <select id="ent-browse">${entBrowseOptions(solid.oid)}</select>
    </div>
    <h3>Geometry</h3>
    <dl>
      <dt>Center</dt><dd>${f3(solid.centroid)}</dd>
      <dt>Size</dt><dd>${f3(size)}</dd>
      <dt>BBox min</dt><dd>${f3(solid.bbox.min)}</dd>
      <dt>BBox max</dt><dd>${f3(solid.bbox.max)}</dd>
    </dl>
    ${solid.pointerErrors.length
      ? `<h3>Pointer errors</h3><div class="hint">${solid.pointerErrors.map(escapeHtml).join('<br>')}</div>`
      : ''}
  `;
  document.getElementById('prop-apply').onclick = () => guard(() => applyProps(solid));
  const subApply = document.getElementById('sub-apply');
  if (subApply) subApply.onclick = () => guard(applySubProps);
  const browse = document.getElementById('ent-browse');
  if (browse) {
    browse.onchange = () => {
      const [type, oid] = browse.value.split(':');
      if (type) selectSub({ type, oid: Number(oid), solid: solid.oid });
    };
  }
}

/* The properties panel's alternative to canvas picking: every surviving
 * face/edge/vertex of the solid, grouped, with the current pick pre-selected. */
function entBrowseOptions(solidOid) {
  const ent = state.entities.find((s) => s.oid === solidOid);
  if (!ent) return '<option value="">(no entity data)</option>';
  const sub = state.subSel;
  const opt = (type, oid, label) => {
    const sel = sub && sub.type === type && sub.oid === oid ? ' selected' : '';
    return `<option value="${type}:${oid}"${sel}>${label}</option>`;
  };
  const groups = [
    ['Faces', ent.faces.map((f) => opt('face', f.oid, `face #${f.oid}${f.nurbs ? ' [NURBS]' : ''}`))],
    ['Edges', ent.edges.map((e) => opt('edge', e, `edge #${e}`))],
    ['Vertices', ent.vertices.map((v) => opt('vertex', v, `vertex #${v}`))],
  ];
  return `<option value="">(pick a face / edge / vertex)</option>`
    + groups.map(([label, opts]) =>
        `<optgroup label="${label} (${opts.length})">${opts.join('')}</optgroup>`).join('');
}

/* ── sub-entity (face/edge/vertex) properties ──────────────────────
 * Read-only geometry/topology rows plus the editable bits: a vertex exposes
 * its position (CLI 'setpoint'), a face/edge exposes its centre (CLI 'move'). */
function subPropsHtml() {
  const info = state.subInfo;
  if (!state.subSel) return '';
  if (!info || info.oid !== state.subSel.oid) {
    return `<h3>${state.subSel.type} #${state.subSel.oid}</h3><div class="hint">loading&hellip;</div>`;
  }
  const f3 = (v) => (v ? v.map((n) => n.toFixed(2)).join(', ') : 'n/a');
  const f = (n) => Number(n.toFixed(3));
  const vec = (idBase, v) => `
    <div class="edit-vec">
      <input id="${idBase}-x" type="number" step="any" value="${f(v[0])}">
      <input id="${idBase}-y" type="number" step="any" value="${f(v[1])}">
      <input id="${idBase}-z" type="number" step="any" value="${f(v[2])}">
    </div>`;
  let rows = '';
  let edit = '';
  if (info.type === 'vertex') {
    rows = `<dt>Position</dt><dd>${f3(info.point)}</dd>`;
    if (info.point) {
      edit = `<div class="edit-grid"><span>Position</span>${vec('sub', info.point)}</div>`;
    }
  } else if (info.type === 'edge') {
    rows = `
      <dt>Curve</dt><dd>${escapeHtml(info.curve)}</dd>
      <dt>Length</dt><dd>${info.length !== null ? f(info.length) : 'n/a'}</dd>
      <dt>Start #${info.aOid ?? '?'}</dt><dd>${f3(info.a)}</dd>
      <dt>End #${info.bOid ?? '?'}</dt><dd>${f3(info.b)}</dd>`;
    if (info.centroid) {
      edit = `<div class="edit-grid"><span>Center</span>${vec('sub', info.centroid)}</div>`;
    }
  } else if (info.type === 'face') {
    rows = `
      <dt>Surface</dt><dd>${escapeHtml(info.surface)}</dd>
      <dt>Loops / Verts</dt><dd>${info.loops} / ${info.numVertices}</dd>
      <dt>Area</dt><dd>${f(info.area)}</dd>
      <dt>Normal</dt><dd>${f3(info.normal)}</dd>
      <dt>Center</dt><dd>${f3(info.centroid)}</dd>`;
    if (info.centroid) {
      edit = `<div class="edit-grid"><span>Center</span>${vec('sub', info.centroid)}</div>`;
    }
  }
  return `
    <h3>${info.type} #${info.oid} <span class="owner">of #${info.solid}</span></h3>
    <dl>${rows}</dl>
    ${edit ? `${edit}<div class="apply-row"><button id="sub-apply" class="primary">Apply ${info.type === 'vertex' ? 'Position' : 'Center'}</button></div>` : ''}
  `;
}

async function applySubProps() {
  const info = state.subInfo;
  if (!info) return;
  const want = ['sub-x', 'sub-y', 'sub-z']
    .map((id) => Number(document.getElementById(id).value));
  if (want.some((n) => !Number.isFinite(n))) throw new Error('coordinates must be numeric');
  if (info.type === 'vertex') {
    await runCommand(`setpoint #${info.oid} as (${want.join(', ')})`);
    return;
  }
  const from = info.centroid;
  const delta = [0, 1, 2].map((i) => want[i] - from[i]);
  if (delta.every((d) => Math.abs(d) < 1e-9)) return;
  await runCommand(`move #${info.oid} ${delta.join(' ')}`);
}

/* Apply the editable properties: rename, and move so the centroid lands on
 * the requested position (both go through the shared kernel). */
async function applyProps(solid) {
  const name = document.getElementById('prop-name').value.trim();
  const want = ['prop-cx', 'prop-cy', 'prop-cz']
    .map((id) => Number(document.getElementById(id).value));
  if (want.some((n) => !Number.isFinite(n))) throw new Error('center must be numeric');

  if (name !== solid.name) {
    const data = await api('/api/rename', { oid: solid.oid, name });
    log(data.message, 'sys');
  }
  const delta = [0, 1, 2].map((i) => want[i] - solid.centroid[i]);
  if (delta.some((d) => Math.abs(d) > 1e-9)) {
    const data = await api('/api/transform', {
      oid: solid.oid, op: 'move', dx: delta[0], dy: delta[1], dz: delta[2],
    });
    log(data.message, 'sys');
  }
  await refreshScene();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function select(oid) {
  state.selected = oid;
  state.subSel = null;
  state.subInfo = null;
  refreshSelectionUi();
}

/* Pick a face/edge/vertex: the owning solid is selected along with it, so the
 * edit menus and solid pickers stay meaningful. */
function selectSub(hit) {
  state.selected = hit.solid;
  state.subSel = { type: hit.type, oid: hit.oid, solid: hit.solid };
  state.subInfo = null;
  refreshSelectionUi();
  guard(async () => {
    const detail = (await api(`/api/entity?oid=${hit.oid}`)).entity;
    // Drop stale responses: the user may have picked something else meanwhile.
    if (state.subSel && state.subSel.oid === hit.oid) {
      state.subInfo = detail;
      renderProps();
    }
  });
}

function refreshSelectionUi() {
  renderSolidList();
  renderProps();
  renderOpParams('sweep');   // entity pickers in the op forms follow the selection
  renderOpParams('combine');
  updateSelStatus();
  draw();
}

function updateSelStatus() {
  const box = document.getElementById('sel-status');
  if (!box) return;
  if (state.subSel) {
    const owner = state.solids.find((s) => s.oid === state.subSel.solid);
    box.textContent = `${state.subSel.type} #${state.subSel.oid} · ${owner ? owner.name : 'solid'} #${state.subSel.solid}`;
    box.classList.add('active');
    box.style.borderColor = TYPE_COLORS[state.subSel.type];
  } else if (state.selected !== null) {
    const solid = state.solids.find((s) => s.oid === state.selected);
    box.textContent = `solid #${state.selected}${solid ? ` '${solid.name}'` : ''}`;
    box.classList.add('active');
    box.style.borderColor = TYPE_COLORS.solid;
  } else {
    box.textContent = 'nothing selected';
    box.classList.remove('active');
    box.style.borderColor = '';
  }
  updateEditHint();
}

/* Spell out what the Edit menu will act on, so the target is never a guess. */
function updateEditHint() {
  const hint = document.getElementById('edit-hint');
  if (!hint) return;
  if (state.subSel) {
    const { type, oid, solid } = state.subSel;
    hint.textContent = `Move/Scale apply to ${type} #${oid}; Rotate/Delete apply to its solid #${solid}.`;
  } else if (state.selected !== null) {
    hint.textContent = `Move/Rotate/Scale/Delete apply to solid #${state.selected}.`;
  } else {
    hint.textContent = 'Select something to edit (canvas click or the Solids list).';
  }
}

/* The canvas badge naming the active pick mode, in that mode's colour. */
function updateModeChip() {
  const chip = document.getElementById('mode-chip');
  if (!chip) return;
  chip.textContent = `pick: ${state.selMode}`;
  chip.style.borderColor = TYPE_COLORS[state.selMode];
  chip.style.color = TYPE_COLORS[state.selMode];
}

/* ── console ───────────────────────────────────────────────────── */
function log(text, cls = '') {
  const pane = document.getElementById('console-log');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = text;
  pane.appendChild(line);
  pane.scrollTop = pane.scrollHeight;
}

async function guard(action) {
  try { await action(); } catch (err) { log(String(err.message || err), 'err'); }
}

/* ── left panel ────────────────────────────────────────────────── */
function renderPrimTabs() {
  const tabs = document.getElementById('prim-tabs');
  tabs.innerHTML = '';
  for (const [kind, def] of Object.entries(PRIMS)) {
    const b = document.createElement('button');
    b.textContent = def.label;
    b.className = kind === state.prim ? 'active' : '';
    b.onclick = () => { state.prim = kind; renderPrimTabs(); renderPrimParams(); };
    tabs.appendChild(b);
  }
}

function renderPrimParams() {
  const box = document.getElementById('prim-params');
  box.innerHTML = '';
  for (const [key, label, value] of PRIMS[state.prim].params) {
    const wrap = document.createElement('label');
    wrap.textContent = label;
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.value = value;
    input.dataset.param = key;
    wrap.appendChild(input);
    box.appendChild(wrap);
  }
}

function primParams() {
  const params = {};
  document.querySelectorAll('#prim-params input').forEach((i) => {
    params[i.dataset.param] = Number(i.value);
  });
  return params;
}

/* ── modeling ops (CLI command builders) ───────────────────────── */
function entityOptions(kind) {
  const opts = [];
  for (const s of state.entities) {
    const tag = `${s.name || 'solid'} #${s.oid}`;
    if (kind === 'solid') {
      opts.push({ value: s.oid, label: tag, solid: s.oid });
      continue;
    }
    if (kind === 'face' || kind === 'entity') {
      for (const f of s.faces) {
        opts.push({
          value: f.oid,
          label: `face #${f.oid}${f.nurbs ? ' [NURBS]' : ''} · ${tag}`,
          solid: s.oid,
        });
      }
    }
    if (kind === 'edge' || kind === 'entity') {
      for (const e of s.edges) {
        opts.push({ value: e, label: `edge #${e} · ${tag}`, solid: s.oid });
      }
    }
  }
  return opts;
}

function renderOpSelect(group) {
  const sel = document.getElementById(`op-select-${group}`);
  sel.innerHTML = '';
  for (const key of OP_GROUPS[group]) {
    sel.appendChild(new Option(OPS[key].label, key));
  }
  sel.value = state.op[group];
  sel.onchange = () => { state.op[group] = sel.value; renderOpParams(group); };
}

function renderOpParams(group) {
  const box = document.getElementById(`op-params-${group}`);
  if (!box) return;
  // Keep whatever the user already typed/picked across scene refreshes.
  const previous = {};
  box.querySelectorAll('[data-param]').forEach((el) => { previous[el.dataset.param] = el.value; });
  box.innerHTML = '';
  const def = OPS[state.op[group]];
  document.getElementById(`op-hint-${group}`).textContent = def.hint;

  for (const field of def.fields) {
    const wrap = document.createElement('label');
    wrap.textContent = field.label;
    let input;
    if (field.type === 'num') {
      input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.value = previous[field.key] !== undefined ? previous[field.key] : field.def;
    } else if (field.type === 'text') {
      input = document.createElement('input');
      input.type = 'text';
      input.spellcheck = false;
      input.value = previous[field.key] !== undefined ? previous[field.key] : (field.def || '');
      wrap.classList.add('span2');
    } else if (field.type === 'select') {
      input = document.createElement('select');
      for (const o of field.options) input.appendChild(new Option(o, o));
      input.value = previous[field.key] !== undefined ? previous[field.key] : field.def;
    } else {
      // solid / face / edge / entity pickers, filled from /api/entities
      input = document.createElement('select');
      const opts = entityOptions(field.type);
      if (!opts.length) {
        input.appendChild(new Option(`(no ${field.type} in the scene)`, ''));
      } else {
        for (const o of opts) input.appendChild(new Option(o.label, o.value));
        const sub = state.subSel;
        const subMatches = sub
          && (field.type === sub.type || (field.type === 'entity' && (sub.type === 'face' || sub.type === 'edge')))
          && opts.some((o) => o.value === sub.oid);
        if (subMatches) {
          input.value = sub.oid;             // face/edge pickers follow the picked entity
        } else if (field.type === 'solid' && state.selected !== null
            && opts.some((o) => o.value === state.selected)) {
          input.value = state.selected;      // solid pickers follow the selection
        } else if (previous[field.key] !== undefined
                   && opts.some((o) => String(o.value) === previous[field.key])) {
          input.value = previous[field.key];
        } else {
          const mine = opts.find((o) => o.solid === state.selected);
          if (mine) input.value = mine.value;
        }
      }
      wrap.classList.add('span2');
    }
    input.dataset.param = field.key;
    wrap.appendChild(input);
    box.appendChild(wrap);
  }
}

function opValues(group) {
  const def = OPS[state.op[group]];
  const values = {};
  for (const field of def.fields) {
    const el = document.querySelector(`#op-params-${group} [data-param="${field.key}"]`);
    const raw = el ? el.value : '';
    if (field.type === 'num') {
      const n = Number(raw);
      if (!Number.isFinite(n)) throw new Error(`'${field.label}' must be a number`);
      values[field.key] = n;
    } else if (field.type === 'text' || field.type === 'select') {
      values[field.key] = raw;
    } else {
      if (raw === '') {
        throw new Error(`No ${field.type} available for '${field.label}' — add or load a solid first.`);
      }
      values[field.key] = Number(raw);
    }
  }
  return values;
}

function selectedOid() {
  if (state.selected === null) throw new Error('Select a solid first.');
  return state.selected;
}

/* Move/Scale honour a picked face/edge/vertex (via the CLI, which transforms
 * any entity); Rotate always spins the owning solid about its centroid. */
async function transform(body) {
  if (state.subSel && body.op === 'move') {
    return runCommand(`move #${state.subSel.oid} ${body.dx} ${body.dy} ${body.dz}`);
  }
  if (state.subSel && body.op === 'scale') {
    return runCommand(`scale #${state.subSel.oid} ${body.factor}`);
  }
  const data = await api('/api/transform', { oid: selectedOid(), ...body });
  log(data.message, 'sys');
  await refreshScene();
}

/* ── splitters ─────────────────────────────────────────────────── */
function wireSplitters() {
  const layout = document.getElementById('layout');
  const MIN = 170;
  const MAX = 520;
  const setup = (id, cssVar, fromLeft) => {
    const bar = document.getElementById(id);
    bar.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      bar.setPointerCapture(event.pointerId);
      bar.classList.add('dragging');
      document.body.classList.add('col-resizing');
      const move = (ev) => {
        const rect = layout.getBoundingClientRect();
        const width = fromLeft
          ? ev.clientX - rect.left
          : rect.right - ev.clientX;
        layout.style.setProperty(cssVar,
          `${Math.max(MIN, Math.min(MAX, Math.round(width)))}px`);
        resize();                      // keep the canvas backing store in sync
      };
      const up = (ev) => {
        bar.releasePointerCapture(ev.pointerId);
        bar.classList.remove('dragging');
        document.body.classList.remove('col-resizing');
        bar.removeEventListener('pointermove', move);
        bar.removeEventListener('pointerup', up);
        resize();
      };
      bar.addEventListener('pointermove', move);
      bar.addEventListener('pointerup', up);
    });
  };
  setup('split-left', '--left-w', true);
  setup('split-right', '--right-w', false);
}

/* ── event wiring ──────────────────────────────────────────────── */
function wire() {
  document.getElementById('btn-deselect').onclick = () => select(null);
  document.getElementById('btn-create').onclick = () => guard(async () => {
    const data = await api('/api/create', { kind: state.prim, params: primParams() });
    log(data.message, 'sys');
    await refreshScene();
    select(data.oid);
    if (state.solids.length === 1) fitView();
  });

  document.getElementById('btn-delete').onclick = () => guard(async () => {
    const data = await api('/api/delete', { oid: selectedOid() });
    log(data.message, 'sys');
    state.selected = null;
    await refreshScene();
  });

  document.getElementById('btn-move').onclick = () => guard(() => transform({
    op: 'move',
    dx: Number(document.getElementById('mv-x').value),
    dy: Number(document.getElementById('mv-y').value),
    dz: Number(document.getElementById('mv-z').value),
  }));

  document.querySelectorAll('[data-nudge]').forEach((button) => {
    button.onclick = () => guard(() => {
      const step = Number(document.getElementById('nudge-step').value) || 1;
      const [x, y, z] = button.dataset.nudge.split(',').map(Number);
      return transform({ op: 'move', dx: x * step, dy: y * step, dz: z * step });
    });
  });

  document.querySelectorAll('#rot-axis button').forEach((button) => {
    button.onclick = () => {
      document.querySelectorAll('#rot-axis button').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
    };
  });
  const rotate = (sign) => guard(() => transform({
    op: 'rotate',
    axis: document.querySelector('#rot-axis button.active').dataset.axis,
    angle: sign * Number(document.getElementById('rot-angle').value),
  }));
  document.getElementById('btn-rot-pos').onclick = () => rotate(1);
  document.getElementById('btn-rot-neg').onclick = () => rotate(-1);

  document.getElementById('btn-scale').onclick = () => guard(() => transform({
    op: 'scale', factor: Number(document.getElementById('scale-factor').value),
  }));

  document.getElementById('btn-save').onclick = () => guard(async () => {
    const data = await api('/api/save', { path: document.getElementById('file-path').value.trim() });
    log(data.message, 'sys');
  });

  document.getElementById('btn-load').onclick = () => guard(async () => {
    const data = await api('/api/load', {
      path: document.getElementById('file-path').value.trim(),
      replace: document.getElementById('load-replace').checked,
    });
    log(data.message, 'sys');
    await refreshScene();
    fitView();
  });

  document.getElementById('btn-refresh').onclick = () => guard(refreshScene);
  document.getElementById('btn-fit').onclick = fitView;

  // Selection mode: which entity type a canvas click picks.
  document.querySelectorAll('#sel-mode button').forEach((button) => {
    button.onclick = () => {
      state.selMode = button.dataset.mode;
      document.querySelectorAll('#sel-mode button').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
      hover = null;
      updateModeChip();
      // Dropping back to solid mode keeps the solid but clears the sub-entity.
      if (state.selMode === 'solid' && state.subSel) {
        state.subSel = null;
        state.subInfo = null;
        refreshSelectionUi();
      } else {
        draw();   // grips appear/disappear with the mode
      }
    };
  });

  // Modeling ops: build the CLI line and run it through the shared shell.
  document.getElementById('btn-op-sweep').onclick =
    () => guard(() => runCommand(OPS[state.op.sweep].build(opValues('sweep'))));
  document.getElementById('btn-op-combine').onclick =
    () => guard(() => runCommand(OPS[state.op.combine].build(opValues('combine'))));

  // Inspect: the same CLI query commands, scoped to the selection when set.
  const selSuffix = () => (state.selected !== null ? ` #${state.selected}` : '');
  document.getElementById('btn-validity').onclick =
    () => guard(() => runCommand(`check validity${selSuffix()}`));
  document.getElementById('btn-topology').onclick =
    () => guard(() => runCommand(`disp topology${selSuffix()}`));
  document.getElementById('btn-vertices').onclick =
    () => guard(() => runCommand(`disp vertices${selSuffix()}`));
  document.getElementById('btn-list').onclick = () => guard(() => runCommand('list'));
  document.getElementById('btn-vars').onclick = () => guard(() => runCommand('vars'));
  document.getElementById('btn-math').onclick = () => guard(() => {
    const ref = document.getElementById('disp-entity').value.trim();
    if (!ref) throw new Error('enter an entity reference (#id, @alias or $var)');
    return runCommand(`disp math ${ref}`);
  });

  document.querySelectorAll('[data-shade]').forEach((button) => {
    button.onclick = () => {
      state.shade = button.dataset.shade;
      document.querySelectorAll('[data-shade]').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
      draw();
    };
  });

  document.getElementById('console-form').onsubmit = (event) => {
    event.preventDefault();
    const input = document.getElementById('console-line');
    const line = input.value.trim();
    if (!line) return;
    input.value = '';
    state.history.push(line);
    state.historyIndex = state.history.length;
    guard(() => runCommand(line));
  };

  document.getElementById('console-line').onkeydown = (event) => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    state.historyIndex += event.key === 'ArrowUp' ? -1 : 1;
    state.historyIndex = Math.max(0, Math.min(state.history.length, state.historyIndex));
    event.target.value = state.history[state.historyIndex] || '';
  };

  // Esc steps the selection back: sub-entity -> its solid -> nothing.
  window.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    const tag = (document.activeElement || {}).tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (state.subSel) select(state.subSel.solid);
    else if (state.selected !== null) select(null);
  });

  wireCanvas();
  wireSplitters();
  window.addEventListener('resize', resize);
}

function wireCanvas() {
  let dragging = false;
  let panning = false;
  let moved = 0;
  let last = [0, 0];

  canvas.addEventListener('pointerdown', (event) => {
    dragging = true;
    panning = event.shiftKey || event.button === 2;
    moved = 0;
    last = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) { updateHover(event); return; }
    const dx = event.clientX - last[0];
    const dy = event.clientY - last[1];
    last = [event.clientX, event.clientY];
    moved += Math.abs(dx) + Math.abs(dy);
    if (moved > 4 && !camDragging) {
      camDragging = true;   // a real drag: hide the grips until pointerup
      hover = null;
    }
    if (panning) {
      const { right, up } = cameraBasis();
      const scale = state.cam.dist * 0.0016;
      for (let i = 0; i < 3; i++) {
        state.cam.target[i] += (-dx * right[i] + dy * up[i]) * scale;
      }
    } else {
      state.cam.yaw -= dx * 0.008;
      state.cam.pitch = Math.max(-1.5, Math.min(1.5, state.cam.pitch + dy * 0.008));
    }
    draw();
  });

  canvas.addEventListener('pointerup', (event) => {
    dragging = false;
    canvas.releasePointerCapture(event.pointerId);
    if (camDragging) {
      camDragging = false;
      draw();                                    // bring the grips back
    }
    if (moved > 4) return;                       // it was a drag, not a click
    const rect = canvas.getBoundingClientRect();
    const hit = pick(event.clientX - rect.left, event.clientY - rect.top);
    if (!hit) {
      select(null);
    } else if (hit.type === 'solid') {
      // Clicking the selected solid again toggles the selection off.
      select(hit.oid === state.selected && !state.subSel ? null : hit.oid);
    } else if (state.subSel && state.subSel.type === hit.type && state.subSel.oid === hit.oid) {
      select(hit.solid);                         // re-click clears the sub-entity
    } else {
      selectSub(hit);
    }
  });

  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    state.cam.dist = Math.max(0.5, state.cam.dist * Math.exp(event.deltaY * 0.0012));
    hover = null;   // screen positions change under the cursor
    draw();
  }, { passive: false });

  canvas.addEventListener('pointerleave', () => {
    if (hover) { hover = null; draw(); }
  });

  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
}

/* Track what is under the cursor and pre-highlight it; the cursor itself
 * signals the mode (crosshair = entity picking) and hover (pointer). */
function updateHover(event) {
  const rect = canvas.getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  const h = pick(px, py);
  canvas.style.cursor = h ? 'pointer' : (state.selMode === 'solid' ? '' : 'crosshair');
  const same = (!h && !hover) || (h && hover && h.type === hover.type && h.oid === hover.oid);
  if (same) return;
  hover = h ? { ...h, px, py } : null;
  draw();
}

/* ── startup ───────────────────────────────────────────────────── */
renderPrimTabs();
renderPrimParams();
renderOpSelect('sweep');
renderOpSelect('combine');
renderOpParams('sweep');
renderOpParams('combine');
updateModeChip();
wire();
resize();
guard(async () => {
  await refreshScene();
  fitView();
  log('B-Rep Authoring Tool ready. It shares the kernel with the CLI session.', 'sys');
});
