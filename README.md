# Advanced CLI B-Rep Kernel Modeler

A text-driven solid modeling engine built **from scratch** (no OpenCascade or any
external CAD library) for CAD research and education. It implements a Boundary
Representation (B-Rep) kernel on a pure half-edge data structure, manipulated only
through atomic **Euler operators**, with a clean MVC separation between topology,
geometry, high-level modeling and the CLI.

<p align="center">
<img src="./doc/img1.png" height="200"></img>
</p>

## Why this exists

Most CAD tutorials either hand-wave the kernel or hide it behind a giant library.
This project keeps the kernel small enough to read end-to-end while staying
topologically rigorous: every operation preserves the Euler–Poincaré invariant

```
V - E + F - R = 2 (S - G)
```

and the validator checks it (plus half-edge pointer consistency) on demand.

## Architecture (strict MVC, layered Model)

```
brep/
  geometry.py    Layer 2  Point3D, Bezier, NURBSSurface, TrimPlane, curve/surface intersection, affine transforms (numpy)
  topology.py    Layer 1  Half-edge entities: Vertex / HalfEdge / Edge / Loop / Face / Solid
  euler_ops.py   Layer 1  Micro Euler operators (mvfs, mev, mef, kef, kemr/mekr, kfmr/mfkr, split_edge)
  macro.py       Layer 3  Macro modeling: extrude, revolve, trim, extend, box/sphere/cylinder/nurbs/plane
  mesh.py        Layer 3  Build a valid half-edge solid from a polygon mesh (sphere/cylinder)
  registry.py    Layer 4  Centralized immutable #id symbol table
  model.py                Kernel: owns the registry + all solids (the "Model")
  validate.py             Euler-Poincare + pointer integrity checks
  view.py        View     Tabular / tree console formatting
  viewer.py      View     3-D interactive viewer (plotly → vedo → tkinter)
  controller.py  Controller  REPL + batch runner (cmd / shlex)
  stepio.py      I/O      STEP (ISO 10303-21) export / best-effort import
main.py                   Entry point
tests/test_kernel.py      Regression tests (assert Euler invariant on many shapes)
scripts/*.bcmd            Example batch command scripts
```

The dependency direction is one-way: Controller → (macro → euler_ops → topology),
geometry, view, validate. Topology never imports geometry's math, and the View
never mutates anything.

## Requirements

* Python 3.10+
* `numpy` (linear algebra and surface evaluation)
* `plotly` (3-D interactive viewer — browser-based, **no numpy required**) — primary viewer
* `vedo` (optional, VTK window) — requires working numpy
* `matplotlib` (optional, fallback wireframe)

```powershell
pip install -r requirements.txt
```

## Running

Interactive REPL:

```powershell
python main.py
```

Run a batch script and drop into the REPL, or run-and-quit with `-q`:

```powershell
python main.py scripts/box.bcmd
python main.py -q scripts/box.bcmd
```

Run the test suite:

```powershell
python tests/test_kernel.py
```

## Quick start

```
brep> create box 10 20 30 as #1
brep> check validity #1
Validity of Solid #1: PASS
  V=8  E=12  F=6  rings=0  shells=1  genus=0
  Euler-Poincare: V-E+F-R = 2  vs  2(S-G) = 2  -> OK
brep> disp topology #1
brep> view @b                    // open 3-D viewer in browser
brep> save "box.step" #1
```

Or build the same box one atomic Euler operator at a time
(each command prints the `#id`s it creates):

```
brep> micro mvfs 0 0 0
brep> micro mev #103 10 0 0
brep> micro mev #105 10 10 0
brep> micro mev #108 0 10 0
brep> micro mef #112 #103
brep> extrude #119 0 0 5
```

## Referring to entities

Macros create many entities whose auto-ids are hard to predict, so three
reference forms are accepted **anywhere a `#id` is expected**:

| Form                                                           | Meaning                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------- |
| `#100` / `100`                                             | a literal id                                                  |
| `@name`                                                      | a user alias, bound with`as @name` or `set @name <token>` |
| `$solid` `$vertex` `$edge` `$face` `$loop` `$last` | the most recently created entity of that kind                 |

The `as` clause on `mvfs`/`mev`/`mef`/`create` accepts either `#id` (request a
specific id) or `@name` (alias the primary entity it makes: solid / edge / face).
`set @name <token>` aliases any existing entity; `vars` lists current aliases and
`$last` values; `unset @name` removes an alias.

```
micro mvfs 0 0 0
set @v0 $vertex          // capture the seed vertex
micro mev $vertex 10 0 0 // grow from the latest vertex each step
micro mev $vertex 10 10 0
micro mev $vertex 0 10 0
micro mef $vertex @v0    // close back to v0
extrude $face 0 0 5      // sweep the face MEF just created
```

This is exactly `scripts/manual_box.bcmd` — it contains no hardcoded ids.

## Command reference

IDs are written with a leading hash, e.g. `#100`. New entities get the next free
id automatically; `as #<id>` requests a specific one, `as @name` aliases it.

In the REPL, `help <topic>` shows detailed usage for any command or primitive.
For example: `help plane`, `help box`, `help trim`, `help create`.

### Micro topology (`micro`)

| Command                                        | Meaning                                        |
| ---------------------------------------------- | ---------------------------------------------- |
| `micro mvfs <x> <y> <z> [as #id]`            | Make Vertex Face Solid – seed a solid         |
| `micro mev #<vertex> <x> <y> <z> [as #edge]` | Make Edge Vertex – grow a wire                |
| `micro mef #<v1> #<v2> [as #face]`           | Make Edge Face – close a loop into a new face |

### Macro modeling

All `create` commands require explicit size parameters. Use `help <kind>` for
parameter details and examples.

| Command                                                          | Required params          | Optional                     | Meaning                                                                                                                                                                                             |
| ---------------------------------------------------------------- | ------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `create box <L> <W> <H>`                                       | L, W, H                  | `[as #id\|@name]`           | axis-aligned box (V=8 E=12 F=6)                                                                                                                                                                     |
| `create sphere <r>`                                            | radius                   | `[slices stacks]` `[as]` | faceted UV sphere                                                                                                                                                                                   |
| `create cylinder <r> <h>`                                      | radius, height           | `[slices]` `[as]`        | capped cylinder                                                                                                                                                                                     |
| `create nurbs <size> <height>`                                 | side length, apex height | `[as]`                     | degree-2 NURBS dome lamina (V=4 E=4 F=2)                                                                                                                                                            |
| `create plane <W> <H>`                                         | width, height            | `[as]`                     | flat rectangular lamina (V=4 E=4 F=2)                                                                                                                                                               |
| `extrude #<face> <dx> <dy> <dz>`                               | face, direction vector   | —                           | sweep face into a prism                                                                                                                                                                             |
| `revolve #<face> <x\|y\|z> <angle>`                              | face, axis, degrees      | `[segments]`               | faceted rotational sweep                                                                                                                                                                            |
| `trim curve #<edge> at <u>`                                    | edge, u∈(0,1)           | —                           | split edge at parameter u (keeps both halves)                                                                                                                                                       |
| `trim surface #<face> keep <u0> <u1> <v0> <v1>`                | face, uv window          | —                           | crop a NURBS face to the kept parametric window                                                                                                                                                     |
| `trim surface #<face> by #<loop>`                              | face, loop id            | —                           | legacy: tag a trim-boundary id (metadata only, no geometry change)                                                                                                                                  |
| `trim #<solid> by plane <nx> <ny> <nz> <d>`                    | solid, plane eq          | `[keep above\|below]`       | geometry-aware half-space cut of any solid (lamina/box/sphere/cylinder/nurbs); cuts edges on their real curve and faces on their real surface;`keep` picks the surviving side (default `above`) |
| `extend #<edge> to plane <nx> <ny> <nz> <d>` \| `to #<face>` | edge, target             | —                           | extend a curve along its tangent until it meets the target (plane / face-plane / NURBS surface); appends a vertex+edge on the contact point                                                         |
| `extend #<face> to plane <nx> <ny> <nz> <d>` \| `to #<face>` | face, target             | `[along <dx> <dy> <dz>]`   | sweep a planar sheet up to the target (extrude-to-target); the new cap lies on the target — flat for a plane, conforming for a NURBS surface                                                       |

### Geometry & editing

| Command                                          | Meaning               |
| ------------------------------------------------ | --------------------- |
| `setpoint #<vertex> as (<x>, <y>, <z>)`        | update coordinates    |
| `setcurve #<edge> as bezier #<cp1> ... #<cpN>` | attach a Bezier curve |
| `setsurface #<face> as nurbs <deg_u> <deg_v>`  | attach a NURBS patch  |

### Transforms

| Command                              | Meaning                          |
| ------------------------------------ | -------------------------------- |
| `move #<entity> <dx> <dy> <dz>`    | translate                        |
| `rotate #<entity> <x\|y\|z> <angle>` | rotate about a principal axis    |
| `scale #<entity> <factor>`         | uniform scale about the centroid |

### Query, I/O & Viewer

| Command                            | Meaning                                                |
| ---------------------------------- | ------------------------------------------------------ |
| `view [solid\|wire\|points] [#id]` | **3-D interactive viewer** (browser/VTK/tkinter) |
| `disp topology [#id]`            | full half-edge pointer tree                            |
| `disp math #<id>`                | underlying equations / control points                  |
| `disp vertices [#id]`            | coordinate table                                       |
| `check validity [#id]`           | Euler + pointer check → PASS / FAIL                   |
| `set @<name> <token>`            | bind a symbolic name to an entity                      |
| `unset @<name>`                  | remove a symbolic name                                 |
| `vars`                           | list aliases and`$last` entities                     |
| `list`                           | summarize all solids                                   |
| `run "<script>"`                 | execute commands from a file                           |
| `save "<file.step>" [#solid]`    | export STEP (AP203 manifold B-rep)                     |
| `load "<file.step>"`             | import CARTESIAN_POINTs                                |
| `exit` / `quit`                | leave the REPL                                         |

In scripts, blank lines and lines starting with `#` or `//` are comments. `//`
also starts an inline comment (`#` cannot, since it collides with `#id`s).

## Example gallery

`scripts/examples.bcmd` builds a cube, sphere, cylinder and NURBS dome, validates
each, and exports them to STEP under `scripts/out/`:

```powershell
python main.py -q scripts/examples.bcmd
```

```
create box 10 10 10 as @cube             -> V=8   E=12  F=6
create sphere 5 24 12 as @sphere         -> V=266 E=552 F=288
create cylinder 4 12 24 as @cyl          -> V=48  E=72  F=26
create nurbs 10 4 as @srf                -> V=4   E=4   F=2  (B-spline surface)
create plane 20 20 as @pl                -> V=4   E=4   F=2  (flat lamina)
```

### 3-D Viewer

The `view` command opens an interactive 3-D viewer.  Backend is auto-detected:

| Backend                    | Install                | Viewer type                                 |
| -------------------------- | ---------------------- | ------------------------------------------- |
| **plotly** (default) | `pip install plotly` | Browser HTML — interactive rotate/zoom/pan |
| **vedo**             | `pip install vedo`   | Native VTK window                           |
| **tkinter**          | stdlib (built-in)      | Perspective wireframe fallback              |

```
brep> create box 10 20 30 as @b
brep> view @b                     // shaded faces + edges (default)
brep> view wire @b                // wireframe only
brep> view points @b              // vertex cloud

brep> create nurbs 20 8 as @nd
brep> trim @nd by plane 1 0 0 0   // topological trim
brep> view @nd                    // NURBS keep-half shown as sampled mesh

brep> create sphere 10 as @sp
brep> view wire @sp               // 128-face sphere wireframe
```

NURBS faces are rendered as a 14×14 sampled surface mesh (orange) alongside
the flat topology faces (blue), so the curved geometry is always visible.

Type `help view` in the REPL for the full reference.

### Trim examples (`scripts/trim_examples.bcmd`)

`scripts/trim_examples.bcmd` runs eight self-contained cases. Each saves a
`NN_<name>_before.step` and a `NN_<name>_after.step` into `scripts/trim_out/`, so
you can open the pair side by side and see exactly what the trim removed:

```powershell
python main.py -q scripts/trim_examples.bcmd
```

Trim is **geometry-aware**: B-Rep carries topology *and* geometry
(`Edge.curve`, `Face.surface`, `Vertex.point`), and a plane cut updates both.
Three regimes result, all of which keep the in-memory topology a valid closed
manifold (so `check validity` passes after every trim) while dropping the discard
half only at export time:

Position the plane so it slices *through* the solid — perpendicular or oblique,
not parallel to a face or grazing it — so the intersection is real and visible.

**1 — Straight/flat topological cut** (plane lamina, box, sphere, cylinder).
`split_edge` inserts the exact intersection vertex on each crossing edge and MEF
splits every straddling face into keep/discard halves. STEP emits
`OPEN_SHELL` + `SHELL_BASED_SURFACE_MODEL` (shell entity referenced per
ISO 10303-42):

```
create box 10 10 10 as @bx
trim @bx by plane 1 1 1 15 keep above     // OBLIQUE cut: lops off the far corner at an angle

create cylinder 6 12 24 as @cy
trim @cy by plane -0.6 0 1 6 keep above   // SLANTED cut z=0.6x+6 -> elliptical top

create sphere 10 as @sp
trim @sp by plane -0.5 0 1 3 keep above    // tilted, off-axis cap (128 -> 64 faces)
```

**2 — Analytic curved cut, boundary crosses** (NURBS dome cut vertically).
The flat boundary *does* cross the plane, so the keep face is MEF-cut and its
NURBS surface is **reparameterized** by De Casteljau subdivision — it still
exports as an analytic `B_SPLINE_SURFACE_WITH_KNOTS` spanning only the keep half:

```
create nurbs 20 8 as @nv
trim @nv by plane 1 0 0 0 keep above      // keep x>0
disp math $face                            // control net cropped to x=[0,10]
save "scripts/trim_out/06_nurbs_vertical_after.step" @nv   // OPEN_SHELL + B_SPLINE_SURFACE
```

**3 — Geometry-aware curved cut, only the surface crosses** (NURBS cap cut).
The dome is a flat lamina (every corner vertex at z=0) carrying a surface that
rises to z≈4. A horizontal plane crosses **no topological edge**, so a
polygon-only trim would miss it entirely. The trim samples the *surface* signed
distance, keeps the curved cap, and exports it as a **faceted `OPEN_SHELL`** that
follows the true surface–plane intersection:

```
create nurbs 20 8 as @cap
trim @cap by plane 0 0 1 3 keep above      // slice the curved cap at z=3
disp math $face                            // "kept +side, faceted along the intersection"
save "scripts/trim_out/07_nurbs_cap_after.step" @cap   // OPEN_SHELL of keep-side triangles
```

An edge that carries a **Bezier curve** is likewise cut on the *curve*: the split
vertex lands on the curve (where it actually meets the plane), not on the
straight chord between its endpoints, and each half keeps its own sub-curve.

**Parametric fallback**: only when the plane *entirely misses* the solid (no edge
and no surface crosses) does the trim store `face.trim_plane` / `face.trim_section`
metadata instead of changing topology; `disp math #<face>` reports the boundary.

**Non-plane trims**: `trim surface #<face> keep <u0> <u1> <v0> <v1>` resizes a
NURBS face to a parametric window (surface is genuinely cropped, still analytic);
`trim curve #<edge> at <u>` splits an edge at `u`, keeping both halves.

Spheres and cylinders are assembled from a polygon mesh via
`brep/mesh.py::build_solid_from_faces` (a closed, consistently-wound manifold),
then pass the same Euler/pointer validation as the Euler-operator shapes.

### Extend examples (`scripts/extend_examples.bcmd`)

`extend` grows a **source** entity until it reaches a **target**, reusing the
same intersection math as trim (line–plane closed form, ray–surface over the
tessellation). Both geometry (a new point *on* the target) and topology (a new
vertex/edge, or a swept strip of faces) are updated. Source and target must be
arranged so the extension path actually **crosses** the target (a right angle or
an oblique angle); a parallel target is never reached, or gives only a trivial
offset. Four source→target combinations are demonstrated; each saves a
before/after pair into `scripts/extend_out/`:

```powershell
python main.py -q scripts/extend_examples.bcmd
```

**Curve → target** (`extend #<edge> to …`): a ray is cast from the edge end along
its tangent, met with the target, and a vertex+edge is appended on the contact
point (Euler +1V +1E). The end whose forward tangent reaches the target is chosen
automatically.

```
// curve -> plane: a wire grows +x into the perpendicular wall x=20
micro mvfs 0 0 0 as @w
micro mev $vertex 6 0 0
extend $edge to plane 1 0 0 20         // hits the wall head-on at (20,0,0)

// curve -> nurbs: a wire below a dome rises +z onto its surface
create nurbs 20 8 as @dome
set @df $face                          // capture the dome's NURBS face
micro mvfs 0 0 -6 as @wire
micro mev $vertex 0 0 -3
extend $edge to @df                    // lands on the dome apex (0,0,4)
```

**Plane → target** (`extend #<face> to …`): the sheet is swept up to the target —
every boundary vertex travels along the sweep direction (default: the face
normal, auto-flipped toward the target) until it meets the target, and side faces
are spun out with MEF, exactly like `extrude` but stopping *on* the target.
Against a **slanted** plane the cap is a ramp; against a NURBS surface each vertex
stops on the surface, so the cap conforms.

```
// plane -> plane: sweep a sheet onto a SLANTED plane -> the cap is a ramp
create plane 20 20 as @s
extend $face to plane -0.5 0 1 10      // z = 10 + 0.5x, not parallel: corners reach z=5..15

// plane -> nurbs: grow a sheet up onto a dome; the cap conforms to the curve
create nurbs 20 8 as @dome
set @df $face
create plane 12 12 as @sheet
move @sheet 0 0 -3
extend $face to @df
```

`extend` never distorts existing geometry — it only appends, so the Euler
invariant is preserved and `check validity` passes after every extend. Type
`help extend` in the REPL for the full reference.

## Verified results

`tests/test_kernel.py` builds boxes, triangular and pentagonal prisms (via both
the macro pipeline and raw Euler operators), a UV sphere, a cylinder and a NURBS
dome, exercises Bezier evaluation/splitting and KEF (the inverse of MEF), and
asserts `V - E + F - R = 2` with no dangling pointers on every shape. The trim
suite covers a lamina plane cut, `keep above`/`keep below`, a NURBS surface-region
crop, the **NURBS cap cut** (surface crosses the plane while the flat boundary
does not — the kept cap tessellates entirely above the plane), and a **curved
edge cut** (a Bezier edge is split *on the curve* at z=3, never on its chord).

## Scope & limitations

This is a teaching kernel, deliberately small:

* **Ring/cavity operators** (`kemr`/`mekr`, `kfmr`/`mfkr`) are implemented as
  library functions for faces-with-holes and handles; the CLI exposes the core
  trio (`mvfs`/`mev`/`mef`) used by the macros.
* **`trim curve`** inserts the split vertex with `split_edge` (both adjacent
  loops updated coherently, no spike) and subdivides the attached Bezier by
  De Casteljau so each half carries its own sub-curve.
* **`trim #<solid> by plane`** performs a *geometry-aware topological trim* for
  **all** solids (lamina, box, sphere, cylinder, NURBS dome): crossing edges are
  split on their real curve, straddling faces MEF-cut, discard faces dropped from
  a STEP `OPEN_SHELL`, and the Euler invariant is preserved throughout. Two
  teaching-scope boundaries remain: (a) an *interior / cap* curved cut — where the
  surface crosses but the flat boundary does not — is kept as a **faceted**
  keep-side shell (a 16×16 tessellation of the true surface–plane intersection),
  not an analytic trimmed B-spline with pcurves, and (b) no cap face is synthesised
  to re-close the cut (the result is an open shell by design). Parametric
  `face.trim_plane` metadata is used only when the plane misses the solid entirely.
* **Surface–surface intersection** (NURBS ∩ NURBS, per the iterative marching
  algorithms in the literature) is not implemented — only plane cutters are.
* **`extend`** grows a curve (tangent ray → contact) or a planar sheet
  (extrude-to-target) onto a plane or NURBS surface, appending topology on the
  computed intersection. The curve extension is a straight tangent continuation
  (C1), not a re-fit Bezier, and a sheet extended onto a curved surface gets a
  faceted conforming cap (its vertices on the surface), not a re-fit NURBS patch.
* **`revolve`** produces a faceted, valid solid with cylindrical-patch tags rather
  than an analytic surface of revolution.
* **STEP import** rebuilds a vertex/point cloud, not the full half-edge graph
  (export is a complete AP203 manifold B-rep).

# License
MIT License

# Author
laputa99999@gmail.com
