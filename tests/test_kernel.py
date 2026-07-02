"""
Smoke / regression tests for the B-Rep kernel.

Run with:  python -m tests.test_kernel   (from the project root)
or simply: python tests/test_kernel.py

Each test asserts the Euler-Poincare invariant and pointer integrity, so a green
run proves the topology stays consistent across the micro and macro operators.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brep import euler_ops as eu
from brep import macro
from brep.geometry import Bezier, Point3D
from brep.model import Kernel
from brep.validate import check_solid


def _assert_valid(solid, label):
    report = check_solid(solid)
    assert report.passed, f"{label} FAILED: euler={report.euler_lhs}=={report.euler_rhs}? " \
                          f"errors={report.pointer_errors}"
    print(f"  [ok] {label}: V={report.v} E={report.e} F={report.f} "
          f"(V-E+F-R={report.euler_lhs})")


def test_macro_box():
    print("test_macro_box")
    k = Kernel()
    s = macro.create_box(k, 10, 20, 30)
    assert (s.num_vertices, s.num_edges, s.num_faces) == (8, 12, 6), \
        f"expected 8/12/6, got {s.num_vertices}/{s.num_edges}/{s.num_faces}"
    _assert_valid(s, "macro box 8V/12E/6F")


def test_manual_box():
    print("test_manual_box")
    k = Kernel()
    solid, face, v0 = eu.mvfs(k, Point3D(0, 0, 0))
    _e, v1 = eu.mev(k, v0, Point3D(10, 0, 0))
    _e, v2 = eu.mev(k, v1, Point3D(10, 10, 0))
    _e, v3 = eu.mev(k, v2, Point3D(0, 10, 0))
    _e, top = eu.mef(k, v3, v0)
    _assert_valid(solid, "square base (4V/4E/2F)")
    assert (solid.num_vertices, solid.num_edges, solid.num_faces) == (4, 4, 2)
    macro.extrude(k, top, Point3D(0, 0, 5))
    assert (solid.num_vertices, solid.num_edges, solid.num_faces) == (8, 12, 6)
    _assert_valid(solid, "manual box after extrude")


def test_triangular_prism():
    print("test_triangular_prism")
    k = Kernel()
    solid, face, v0 = eu.mvfs(k, Point3D(0, 0, 0))
    _e, v1 = eu.mev(k, v0, Point3D(10, 0, 0))
    _e, v2 = eu.mev(k, v1, Point3D(5, 8, 0))
    _e, top = eu.mef(k, v2, v0)
    macro.extrude(k, top, Point3D(0, 0, 4))
    # triangle prism: V=6, E=9, F=5
    assert (solid.num_vertices, solid.num_edges, solid.num_faces) == (6, 9, 5), \
        f"got {solid.num_vertices}/{solid.num_edges}/{solid.num_faces}"
    _assert_valid(solid, "triangular prism (6V/9E/5F)")


def test_pentagon_prism():
    print("test_pentagon_prism")
    import math
    k = Kernel()
    pts = [Point3D(math.cos(math.radians(72 * i)), math.sin(math.radians(72 * i)), 0)
           for i in range(5)]
    solid, face, v0 = eu.mvfs(k, pts[0])
    prev = v0
    for p in pts[1:]:
        _e, prev = eu.mev(k, prev, p)
    _e, top = eu.mef(k, prev, v0)
    macro.extrude(k, top, Point3D(0, 0, 3))
    # pentagon prism: V=10, E=15, F=7
    assert (solid.num_vertices, solid.num_edges, solid.num_faces) == (10, 15, 7), \
        f"got {solid.num_vertices}/{solid.num_edges}/{solid.num_faces}"
    _assert_valid(solid, "pentagon prism (10V/15E/7F)")


def test_bezier_geometry():
    print("test_bezier_geometry")
    curve = Bezier([Point3D(0, 0, 0), Point3D(1, 2, 0), Point3D(2, 2, 0), Point3D(3, 0, 0)])
    mid = curve.evaluate(0.5)
    assert abs(mid.x - 1.5) < 1e-9, mid
    left, right = curve.split(0.5)
    assert left.control_points[0].is_close(Point3D(0, 0, 0))
    assert right.control_points[-1].is_close(Point3D(3, 0, 0))
    print("  [ok] bezier evaluate/split")


def test_trim_curve():
    print("test_trim_curve")
    k = Kernel()
    solid, face, v0 = eu.mvfs(k, Point3D(0, 0, 0))
    edge, v1 = eu.mev(k, v0, Point3D(10, 0, 0))
    edge.curve = Bezier([Point3D(0, 0, 0), Point3D(3, 5, 0),
                         Point3D(7, 5, 0), Point3D(10, 0, 0)])
    v_before, e_before = solid.num_vertices, solid.num_edges

    new_v, new_e = macro.trim_curve(k, edge, 0.5)

    # Topological split, not a dangling spike: +1V, +1E.
    assert (solid.num_vertices, solid.num_edges) == (v_before + 1, e_before + 1), \
        f"expected +1V/+1E, got {solid.num_vertices}/{solid.num_edges}"
    # The two resulting segments must be exactly A↔M and M↔B (a spike would leave
    # the original A↔B edge intact and dangle a separate A↔M off vertex A).
    seg = lambda e: frozenset((e.he1.vertex, e.he1.end_vertex))
    assert {seg(edge), seg(new_e)} == {frozenset((v0, new_v)), frozenset((new_v, v1))}, \
        "edge must split into connected A-M and M-B segments (no spike)"
    # Each segment's Bezier must span its own topological endpoints.
    assert edge.curve is not None and new_e.curve is not None
    for e in (edge, new_e):
        a, b = e.he1.vertex, e.he1.end_vertex
        assert (e.curve.evaluate(0.0) - a.point).length() < 1e-9
        assert (e.curve.evaluate(1.0) - b.point).length() < 1e-9
    print(f"  [ok] trim split vertex at {new_v.point} (V={solid.num_vertices} "
          f"E={solid.num_edges})")


def test_sphere():
    print("test_sphere")
    k = Kernel()
    slices, stacks = 12, 6
    s = macro.create_sphere(k, 5.0, slices, stacks)
    exp_v = 2 + (stacks - 1) * slices
    exp_f = stacks * slices
    assert (s.num_vertices, s.num_faces) == (exp_v, exp_f), \
        f"got {s.num_vertices}/{s.num_faces}, expected {exp_v}/{exp_f}"
    _assert_valid(s, f"UV sphere ({slices}x{stacks})")


def test_cylinder():
    print("test_cylinder")
    k = Kernel()
    slices = 16
    s = macro.create_cylinder(k, 4.0, 10.0, slices)
    assert (s.num_vertices, s.num_faces) == (2 * slices, slices + 2), \
        f"got {s.num_vertices}/{s.num_faces}"
    _assert_valid(s, f"cylinder ({slices})")


def test_nurbs_dome():
    print("test_nurbs_dome")
    from brep.geometry import NURBSSurface
    k = Kernel()
    s = macro.create_nurbs_dome(k, 10.0, 4.0)
    _assert_valid(s, "nurbs dome lamina (4V/4E/2F)")
    front = next(f for f in s.faces if f.surface is not None)
    assert isinstance(front.surface, NURBSSurface)
    apex = front.surface.evaluate(0.5, 0.5)
    assert apex.z > 1.0, f"dome apex should rise, got {apex}"
    print(f"  [ok] dome apex at {apex}")


def test_trim_solid_by_plane():
    print("test_trim_solid_by_plane")
    # Volumetric solids now receive a REAL topological trim (not metadata).
    # (a) sphere sliced off-vertex at z=3 → crossing edges are split.
    k = Kernel()
    s = macro.create_sphere(k, 10.0, 12, 6)
    v0, e0 = s.num_vertices, s.num_edges
    res = macro.trim_solid_by_plane(k, s, 0, 0, 1, 3)
    assert res.is_topological, "solid trim must be topological, not metadata"
    assert s.num_vertices > v0 and s.num_edges > e0, "crossing edges must split"
    assert res.n_cut > 0 and res.n_discard > 0
    _assert_valid(s, "sphere trimmed at z=3")
    kept = [f for f in s.faces if not getattr(f, "discarded", False)]
    assert all(  # every surviving face lies on/above the cut plane
        min(he.vertex.point.z for he in f.outer.halfedges()) > 3 - 1e-6
        for f in kept
    ), "a discarded face survived the trim"

    # (b) clean cut through the equator ring (z=0) needs no edge split.
    k = Kernel()
    s = macro.create_sphere(k, 10.0, 12, 6)
    v0 = s.num_vertices
    res = macro.trim_solid_by_plane(k, s, 0, 0, 1, 0)
    assert res.is_topological and res.n_cut == 0, "equator cut should not split edges"
    assert s.num_vertices == v0, "through-vertex cut must not add vertices"
    assert res.n_keep == 36 and res.n_discard == 36, \
        f"upper/lower hemisphere split, got keep={res.n_keep} discard={res.n_discard}"
    _assert_valid(s, "sphere sliced at equator (z=0)")
    print("  [ok] sphere trims topologically (offset + equator)")


def test_trim_keep_side():
    print("test_trim_keep_side")
    # 'keep below' must retain the complementary half of 'keep above' — same
    # reference plane, opposite side. Cut a 20x20 plane lamina at y=3.
    def kept_y_range(keep_below):
        k = Kernel()
        s = macro.create_plane(k, 20, 20)
        macro.trim_solid_by_plane(k, s, 0, 1, 0, 3, keep_below=keep_below)
        ys = [he.vertex.point.y
              for f in s.faces if not getattr(f, "discarded", False)
              for he in f.outer.halfedges()]
        return min(ys), max(ys)

    above = kept_y_range(False)
    below = kept_y_range(True)
    assert above == (3.0, 10.0), f"keep above should be y in [3,10], got {above}"
    assert below == (-10.0, 3.0), f"keep below should be y in [-10,3], got {below}"
    print(f"  [ok] keep above -> y{above}, keep below -> y{below}")


def test_trim_surface_region():
    print("test_trim_surface_region")
    from brep.geometry import NURBSSurface
    k = Kernel()
    s = macro.create_nurbs_dome(k, 20.0, 8.0)
    face = next(f for f in s.faces if isinstance(f.surface, NURBSSurface))
    full = face.surface
    # The cropped surface's corners must equal the full surface at the window.
    u0, u1, v0, v1 = 0.25, 0.75, 0.25, 0.75
    expect = [full.evaluate(u, v) for u in (u0, u1) for v in (v0, v1)]
    macro.trim_surface_region(k, face, u0, u1, v0, v1)
    assert isinstance(face.surface, NURBSSurface)
    got = [face.surface.evaluate(u, v) for u in (0.0, 1.0) for v in (0.0, 1.0)]
    for e, g in zip(expect, got):
        assert (e - g).length() < 1e-6, f"crop corner mismatch: {e} vs {g}"
    assert getattr(face, "trim_uv", None) == (u0, u1, v0, v1)
    print(f"  [ok] NURBS face cropped to u=[{u0},{u1}] v=[{v0},{v1}]")


def test_trim_nurbs_cap_by_plane():
    print("test_trim_nurbs_cap_by_plane")
    from brep.geometry import NURBSSurface, TrimPlane, tessellate_surface_trim
    # A dome is a FLAT lamina (all boundary verts at z=0) carrying a curved NURBS
    # surface. A horizontal plane z>2 slices the curved cap but crosses NO
    # topological edge — the polygon-only trim used to miss this entirely.
    k = Kernel()
    d = macro.create_nurbs_dome(k, 20.0, 8.0)
    res = macro.trim_solid_by_plane(k, d, 0, 0, 1, 2.0)
    assert res.is_topological, "curved cap cut must be a real (topological) trim"
    _assert_valid(d, "dome after cap trim z>2")
    # The curved (NURBS) face survives and is tagged with the trim plane; the
    # flat underside is discarded.
    nurbs_faces = [f for f in d.faces if isinstance(f.surface, NURBSSurface)]
    kept = [f for f in nurbs_faces if not getattr(f, "discarded", False)]
    assert kept, "the curved cap face must be kept, not discarded"
    plane = kept[0].trim_plane
    assert isinstance(plane, TrimPlane)
    # The faceted keep-side mesh must lie entirely on the +side of the plane.
    pts, tris = tessellate_surface_trim(kept[0].surface, plane, 12, 12)
    assert tris, "kept cap must tessellate to a non-empty mesh"
    assert all(plane.signed_distance(p) >= -1e-6 for p in pts), \
        "trimmed cap mesh leaked below the cut plane"
    print(f"  [ok] dome cap kept as {len(tris)}-triangle mesh, all above z=2")


def test_trim_edge_follows_curve():
    print("test_trim_edge_follows_curve")
    from brep.geometry import Bezier, TrimPlane, bezier_plane_param
    # An edge carrying a Bezier that bulges up in z. A plane the straight chord
    # never reaches must still cut the edge — on the CURVE, not the chord.
    k = Kernel()
    s = macro.create_plane(k, 20, 20)
    e = s.edges[0]
    a = e.he1.vertex.point
    b = e.he1.end_vertex.point
    e.curve = Bezier([a, (a + b) * 0.5 + Point3D(0, 0, 16), b])  # peak z=8
    plane = TrimPlane(Point3D(0, 0, 1), 3.0)
    u = bezier_plane_param(e.curve, plane)
    assert u is not None and 0.0 < u < 1.0
    on_curve = e.curve.evaluate(u)
    assert abs(on_curve.z - 3.0) < 1e-6, "crossing must be found ON the curve"
    n_before = s.num_vertices
    macro.trim_solid_by_plane(k, s, 0, 0, 1, 3.0)
    assert s.num_vertices == n_before + 1, "curved edge must be split once"
    _assert_valid(s, "lamina after curved-edge trim")
    # The inserted vertex sits on the curve (z=3), never on the flat chord (z=0).
    assert any(abs(v.point.z - 3.0) < 1e-6 for v in s.vertices), \
        "split vertex must lie on the curve, not the straight chord"
    print("  [ok] curved edge split on the curve at z=3 (not the chord)")


def test_extend_curve_to_plane():
    print("test_extend_curve_to_plane")
    from brep.geometry import TrimPlane
    # A lamina boundary edge extended to plane y=25: the new vertex lands exactly
    # on the plane and topology grows by one vertex + one edge.
    k = Kernel()
    s = macro.create_plane(k, 20, 20)
    edge = s.edges[0]                      # bottom edge, tangent along +/-x... pick reachable
    plane = TrimPlane(Point3D(1, 0, 0), 25.0)   # bottom edge tangent is +/-x -> reaches x=25
    nv0 = s.num_vertices
    new_v, new_e = macro.extend_curve(k, edge, ("plane", plane))
    assert abs(plane.signed_distance(new_v.point)) < 1e-6, "extended vertex must lie on the plane"
    assert s.num_vertices == nv0 + 1 and new_e is not None
    _assert_valid(s, "lamina after edge extend to plane")
    print(f"  [ok] edge extended to plane, new vertex on plane at {new_v.point}")


def test_extend_curve_to_surface():
    print("test_extend_curve_to_surface")
    from brep.geometry import NURBSSurface
    # A vertical wire extended up onto a dome: the new vertex lands ON the surface.
    k = Kernel()
    dome = macro.create_nurbs_dome(k, 20.0, 8.0)
    surf = next(f.surface for f in dome.faces if isinstance(f.surface, NURBSSurface))
    _solid, _f, v0 = eu.mvfs(k, Point3D(0, 0, -5))
    edge, _v1 = eu.mev(k, v0, Point3D(0, 0, -3))     # wire pointing +z
    new_v, _new_e = macro.extend_curve(k, edge, ("surface", surf))
    # apex of this dome is z=4 at (0,0); the +z ray must hit the surface there.
    assert (new_v.point - Point3D(0, 0, 4)).length() < 1e-3, \
        f"extended vertex must land on the dome apex, got {new_v.point}"
    print(f"  [ok] wire extended onto NURBS surface at {new_v.point}")


def test_extend_face_to_plane():
    print("test_extend_face_to_plane")
    from brep.geometry import TrimPlane
    # A flat sheet swept up to z=10 becomes a closed box; the cap lies on z=10.
    k = Kernel()
    s = macro.create_plane(k, 20, 20)
    face = s.faces[0]
    macro.extend_face(k, face, ("plane", TrimPlane(Point3D(0, 0, 1), 10.0)))
    _assert_valid(s, "sheet extended to plane (box)")
    zs = sorted({round(v.point.z, 6) for v in s.vertices})
    assert zs == [0.0, 10.0], f"cap must reach z=10, got {zs}"
    assert (s.num_vertices, s.num_edges, s.num_faces) == (8, 12, 6)
    print("  [ok] sheet extended to a box, cap on z=10")


def test_extend_face_to_surface():
    print("test_extend_face_to_surface")
    from brep.geometry import NURBSSurface, TrimPlane
    # A flat sheet below a dome, swept up onto the curved surface: every cap
    # vertex must land on the surface (its ray hit), and topology stays valid.
    k = Kernel()
    dome = macro.create_nurbs_dome(k, 20.0, 8.0)
    surf = next(f.surface for f in dome.faces if isinstance(f.surface, NURBSSurface))
    sheet = macro.create_plane(k, 12, 12, origin=Point3D(0, 0, -3))
    face = sheet.faces[0]
    macro.extend_face(k, face, ("surface", surf))
    _assert_valid(sheet, "sheet extended to NURBS surface")
    # Cap vertices (z > -3) must sit on the surface: a +z ray from each hits it.
    from brep.geometry import ray_surface_intersect
    cap = [v.point for v in sheet.vertices if v.point.z > -3 + 1e-6]
    assert cap, "expected a swept cap above the base"
    for p in cap:
        hit = ray_surface_intersect(Point3D(p.x, p.y, -5), Point3D(0, 0, 1), surf)
        assert hit is not None and abs(hit.z - p.z) < 1e-3, \
            f"cap vertex {p} is not on the surface"
    print(f"  [ok] sheet cap conforms to the dome ({len(cap)} vertices on surface)")


def test_kef_inverse_of_mef():
    print("test_kef_inverse_of_mef")
    k = Kernel()
    solid, face, v0 = eu.mvfs(k, Point3D(0, 0, 0))
    _e, v1 = eu.mev(k, v0, Point3D(10, 0, 0))
    _e, v2 = eu.mev(k, v1, Point3D(10, 10, 0))
    _e, v3 = eu.mev(k, v2, Point3D(0, 10, 0))
    closing_edge, new_face = eu.mef(k, v3, v0)
    assert (solid.num_vertices, solid.num_edges, solid.num_faces) == (4, 4, 2)
    # KEF removes the edge MEF added, merging the two faces back into one.
    eu.kef(k, closing_edge)
    assert (solid.num_vertices, solid.num_edges, solid.num_faces) == (4, 3, 1), \
        f"got {solid.num_vertices}/{solid.num_edges}/{solid.num_faces}"
    _assert_valid(solid, "after KEF (4V/3E/1F)")


def run_all():
    tests = [
        test_macro_box,
        test_manual_box,
        test_triangular_prism,
        test_pentagon_prism,
        test_bezier_geometry,
        test_trim_curve,
        test_sphere,
        test_cylinder,
        test_nurbs_dome,
        test_trim_solid_by_plane,
        test_trim_keep_side,
        test_trim_surface_region,
        test_trim_nurbs_cap_by_plane,
        test_trim_edge_follows_curve,
        test_extend_curve_to_plane,
        test_extend_curve_to_surface,
        test_extend_face_to_plane,
        test_extend_face_to_surface,
        test_kef_inverse_of_mef,
    ]
    for t in tests:
        t()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
