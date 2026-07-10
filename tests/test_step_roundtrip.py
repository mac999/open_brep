"""
STEP round-trip + authoring web app regression tests.

Run with:  python -m pytest tests/test_step_roundtrip.py
or simply: python tests/test_step_roundtrip.py

The point of these: a shape authored in the web app is only useful to the CLI if
``save`` -> ``load`` gives back *topology*, not a bag of points. Each test below
asserts the reconstructed solid matches the original's V/E/F and still passes the
Euler-Poincare and pointer-integrity checks.
"""

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brep import macro, stepio
from brep.geometry import NURBSSurface
from brep.model import Kernel
from brep.validate import check_solid


def _roundtrip(solids, faceted=False):
    """Save ``solids`` to a temp STEP file and load it back into a fresh kernel."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scene.step")
        stepio.save_solids(solids, path, faceted=faceted)
        return stepio.load(Kernel(), path)


def test_roundtrip_primitives_preserve_topology():
    print("test_roundtrip_primitives_preserve_topology")
    k = Kernel()
    originals = [
        macro.create_box(k, 10, 20, 30),
        macro.create_sphere(k, 5),
        macro.create_cylinder(k, 4, 12),
        macro.create_plane(k, 20, 20),
    ]
    restored = _roundtrip(originals)
    assert len(restored) == len(originals), "one solid per shell expected"
    for before, after in zip(originals, restored):
        report = check_solid(after)
        assert report.passed, f"{before.name}: {report.pointer_errors}"
        assert (after.num_vertices, after.num_edges, after.num_faces) == \
               (before.num_vertices, before.num_edges, before.num_faces)
        assert after.name == before.name, "the shell's name should survive"
        print(f"  [ok] {before.name}: V={after.num_vertices} E={after.num_edges} "
              f"F={after.num_faces} rebuilt and valid")


def test_roundtrip_restores_nurbs_surface():
    print("test_roundtrip_restores_nurbs_surface")
    k = Kernel()
    dome = macro.create_nurbs_dome(k, 20.0, 8.0)
    source = next(f.surface for f in dome.faces if f.surface)

    restored = _roundtrip([dome])[0]
    surfaces = [f.surface for f in restored.faces
                if isinstance(f.surface, NURBSSurface)]
    assert len(surfaces) == 1, "the dome's single B-spline face should come back"
    surf = surfaces[0]
    assert (surf.n_u, surf.n_v) == (source.n_u, source.n_v)
    assert (surf.degree_u, surf.degree_v) == (source.degree_u, source.degree_v)

    # The surface must evaluate to the same points, not merely have the same shape.
    for u, v in ((0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (1.0, 1.0)):
        a, b = source.evaluate(u, v), surf.evaluate(u, v)
        assert (a - b).length() < 1e-5, f"S({u},{v}) drifted: {a} vs {b}"
    print(f"  [ok] {surf.n_u}x{surf.n_v} degree-{surf.degree_u} B-spline restored; "
          "apex and corners evaluate identically")


def test_roundtrip_trimmed_open_shell():
    print("test_roundtrip_trimmed_open_shell")
    k = Kernel()
    plane = macro.create_plane(k, 20, 20)
    macro.trim_solid_by_plane(k, plane, 0, 1, 0, 3.0)
    kept = [f for f in plane.faces if not getattr(f, "discarded", False)]

    restored = _roundtrip([plane])[0]
    # Only the surviving half is written, so that is all that comes back.
    assert restored.num_faces == len(kept) == 2
    ys = [v.point.y for v in restored.vertices]
    assert min(ys) >= 3.0 - 1e-6, "the discarded half must not be re-imported"
    print(f"  [ok] trimmed lamina: {restored.num_faces} keep faces, "
          f"y >= {min(ys):.1f} (the cut)")


def test_load_falls_back_to_points_for_untopological_step():
    print("test_load_falls_back_to_points_for_untopological_step")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cloud.step")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("ISO-10303-21;\nDATA;\n"
                     "#1 = CARTESIAN_POINT('',(0.,0.,0.));\n"
                     "#2 = CARTESIAN_POINT('',(1.,2.,3.));\n"
                     "ENDSEC;\nEND-ISO-10303-21;\n")
        solids = stepio.load(Kernel(), path)
    assert len(solids) == 1 and solids[0].num_faces == 0
    assert solids[0].num_vertices == 2
    print("  [ok] a file with no ADVANCED_FACE degrades to a 2-point vertex cloud")


def test_faceted_export_roundtrips_as_a_shell():
    print("test_faceted_export_roundtrips_as_a_shell")
    k = Kernel()
    dome = macro.create_nurbs_dome(k, 20.0, 8.0)
    macro.trim_solid_by_plane(k, dome, 0, 0, 1, 3.0)   # cap ring trim
    restored = _roundtrip([dome], faceted=True)[0]
    assert restored.num_faces > 50, "the triangle shell should rebuild face by face"
    zs = [v.point.z for v in restored.vertices]
    assert min(zs) >= 3.0 - 1e-3, "faceted export keeps only the z>3 cap"
    print(f"  [ok] faceted cap: {restored.num_faces} triangles, z >= {min(zs):.2f}")


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #
class _Client:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as response:
            return json.load(response)

    def post(self, path, body):
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            return json.load(exc)


def test_webapp_shares_the_shell_kernel():
    print("test_webapp_shares_the_shell_kernel")
    from brep import webapp
    from brep.controller import BRepShell

    shell = BRepShell()
    client = _Client(webapp.start(shell, port=0, open_browser=False))
    try:
        created = client.post("/api/create",
                              {"kind": "box", "params": {"length": 10, "width": 10,
                                                         "height": 10}})
        assert created["ok"], created
        oid = created["oid"]
        # The browser's box is the REPL's box: same kernel, same registry.
        assert [s.oid for s in shell.kernel.solids] == [oid]

        scene = client.get("/api/scene")["solids"][0]
        assert scene["stats"]["f"] == 6 and scene["valid"]
        assert len(scene["triangles"]) == 12, "6 quads fan-triangulate to 12 tris"

        moved = client.post("/api/transform",
                            {"oid": oid, "op": "move", "dx": 5, "dy": 0, "dz": 0})
        assert moved["ok"], moved
        assert shell.kernel.solids[0].vertices[0].point.x == 5.0

        # A command typed in the browser console runs in the same shell.
        out = client.post("/api/command", {"line": "create sphere 3 as @s"})
        assert out["ok"] and "s" in shell.aliases   # aliases are keyed without '@'
        assert len(shell.kernel.solids) == 2

        # ... but the commands that would hang or kill the server are refused.
        for banned in ("exit", "view", "webapp"):
            assert not client.post("/api/command", {"line": banned})["ok"]

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scene.step").replace("\\", "/")
            assert client.post("/api/save", {"path": path})["ok"]
            loaded = client.post("/api/load", {"path": path, "replace": True})
            assert loaded["ok"] and len(loaded["oids"]) == 2
            # 'replace' cleared the old solids; only the imported ones remain.
            assert [s.oid for s in shell.kernel.solids] == loaded["oids"]

        assert client.post("/api/delete", {"oid": loaded["oids"][0]})["ok"]
        assert len(shell.kernel.solids) == 1

        assert not client.post("/api/create", {"kind": "torus"})["ok"]
        print("  [ok] create/transform/command/save/load/delete all hit one kernel")
    finally:
        webapp.stop()


def test_webapp_entities_and_op_menus_reuse_the_cli():
    print("test_webapp_entities_and_op_menus_reuse_the_cli")
    from brep import webapp
    from brep.controller import BRepShell

    shell = BRepShell()
    client = _Client(webapp.start(shell, port=0, open_browser=False))
    try:
        assert client.post("/api/command", {"line": "create nurbs 20 8 as @n"})["ok"]
        assert client.post("/api/command",
                           {"line": "create box 8 8 8 as @b"})["ok"]

        # /api/entities feeds the op menus' id pickers with surviving ids only.
        ents = client.get("/api/entities")
        assert ents["ok"] and len(ents["solids"]) == 2
        dome, box = ents["solids"]
        assert any(f["nurbs"] for f in dome["faces"])
        assert len(box["faces"]) == 6 and len(box["edges"]) == 12
        assert len(box["vertices"]) == 8

        # The modeling-op menus send plain CLI lines through /api/command; the
        # ids they insert come from /api/entities.
        nurbs_face = next(f["oid"] for f in dome["faces"] if f["nurbs"])
        box_oid, box_edge = box["oid"], box["edges"][0]
        for line in (
            f"trim #{box_oid} by plane 1 1 1 15 keep below",
            f"trim curve #{box_edge} at 0.5",
            "create plane 12 12 as @sheet",
            "move @sheet 0 0 -3",
            f"extend $face to #{nurbs_face}",
            "check validity",
            "disp topology",
        ):
            result = client.post("/api/command", {"line": line})
            assert result["ok"], (line, result)

        # After the trim, /api/entities reflects the new surviving topology
        # (split vertices added, the discarded corner's ids gone).
        box_after = next(s for s in client.get("/api/entities")["solids"]
                         if s["oid"] == box_oid)
        assert box_after["vertices"] != box["vertices"]
        assert {f["oid"] for f in box_after["faces"]} != {f["oid"] for f in box["faces"]}
        print("  [ok] entities picker + op menus drive the shared CLI dispatch")
    finally:
        webapp.stop()


def test_webapp_picking_and_entity_props():
    print("test_webapp_picking_and_entity_props")
    from brep import webapp
    from brep.controller import BRepShell

    shell = BRepShell()
    client = _Client(webapp.start(shell, port=0, open_browser=False))
    try:
        assert client.post("/api/command", {"line": "create box 10 10 10 as @b"})["ok"]

        # The scene carries the picking metadata the canvas needs: the owning
        # face per triangle, the edge per wire segment, and vertex positions.
        scene = client.get("/api/scene")["solids"][0]
        assert len(scene["triFaces"]) == len(scene["triangles"])
        assert len(scene["wireEdges"]) == len(scene["wire"])
        assert len(scene["verts"]) == scene["stats"]["v"]
        assert len(scene["faceGrips"]) == scene["stats"]["f"]
        assert all(len(g["p"]) == 3 for g in scene["faceGrips"])

        # NURBS grips are evaluated on the surface, and the whole scene stays
        # JSON-serializable with a NURBS solid present.
        assert client.post("/api/command", {"line": "create nurbs 20 8 as @n"})["ok"]
        dome = client.get("/api/scene")["solids"][1]
        assert dome["faceGrips"] and all(
            isinstance(c, float) for g in dome["faceGrips"] for c in g["p"])
        assert any(g["p"][2] > 1.0 for g in dome["faceGrips"]), \
            "the dome's surface grip should sit up on the bulge, not the base"

        ents = client.get("/api/entities")["solids"][0]
        face, edge, vert = ents["faces"][0]["oid"], ents["edges"][0], ents["vertices"][0]

        # /api/entity backs the right-hand props panel per entity type.
        fd = client.get(f"/api/entity?oid={face}")["entity"]
        assert fd["type"] == "face" and fd["surface"] == "plane"
        assert abs(fd["area"] - 100.0) < 1e-6 and fd["loops"] == 1
        ed = client.get(f"/api/entity?oid={edge}")["entity"]
        assert ed["type"] == "edge" and ed["curve"] == "line"
        assert abs(ed["length"] - 10.0) < 1e-9
        vd = client.get(f"/api/entity?oid={vert}")["entity"]
        assert vd["type"] == "vertex" and len(vd["point"]) == 3

        # The panel's editable fields go through the shared CLI: a vertex
        # position via 'setpoint', a face centre via a 'move' delta.
        assert client.post("/api/command",
                           {"line": f"setpoint #{vert} as (1, 2, 3)"})["ok"]
        assert client.get(f"/api/entity?oid={vert}")["entity"]["point"] == [1, 2, 3]
        before = client.get(f"/api/entity?oid={face}")["entity"]["centroid"]
        assert client.post("/api/command", {"line": f"move #{face} 0 0 5"})["ok"]
        after = client.get(f"/api/entity?oid={face}")["entity"]["centroid"]
        assert abs(after[2] - before[2] - 5.0) < 1e-9

        # Only sub-entities have a detail view; a solid id is a client error.
        try:
            urllib.request.urlopen(f"{client.base}/api/entity?oid={ents['oid']}")
            raise AssertionError("solid detail should 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

        # A trim leaves discarded faces in solid.faces; asking for their
        # detail must be a clean 400 (no boundary to mesh), not a 500.
        assert client.post("/api/command",
                           {"line": f"trim #{ents['oid']} by plane 0 0 1 5 keep below"})["ok"]
        dead = next(f.oid for f in shell.kernel.solids[0].faces
                    if getattr(f, "discarded", False) or f.outer is None)
        try:
            urllib.request.urlopen(f"{client.base}/api/entity?oid={dead}")
            raise AssertionError("discarded face detail should 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        print("  [ok] picking metadata + face/edge/vertex props + edits via the CLI")
    finally:
        webapp.stop()


def test_webapp_serves_its_own_assets():
    print("test_webapp_serves_its_own_assets")
    from brep import webapp
    from brep.controller import BRepShell

    base = webapp.start(BRepShell(), port=0, open_browser=False).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/") as response:
            assert b"<canvas" in response.read()
        for asset in ("/static/app.js", "/static/styles.css"):
            with urllib.request.urlopen(base + asset) as response:
                assert len(response.read()) > 500, asset
        # No directory traversal out of brep/web/.
        try:
            urllib.request.urlopen(base + "/static/../controller.py")
            raise AssertionError("traversal should 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        print("  [ok] index + assets served, traversal refused, no CDN needed")
    finally:
        webapp.stop()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall step round-trip + webapp tests passed")
