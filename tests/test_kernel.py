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


def test_numeric_intersection():
    print("test_numeric_intersection")
    from brep.geometry import (NURBSSurface, TrimPlane, surface_closest_point,
                               ray_surface_intersect_ex, surface_plane_section)
    k = Kernel()
    d = macro.create_nurbs_dome(k, 20.0, 8.0)
    surf = next(f.surface for f in d.faces if isinstance(f.surface, NURBSSurface))
    # Closest-point projection (Beer Alg. 2): a point above the apex projects to it.
    u, v, foot = surface_closest_point(surf, Point3D(0, 0, 10))
    assert abs(u - 0.5) < 1e-6 and abs(v - 0.5) < 1e-6
    assert (foot - Point3D(0, 0, 4)).length() < 1e-9
    # Ray refinement (Beer Alg. 3): the contact lies on the TRUE surface, not a
    # facet (facet-only accuracy was ~3e-3); and exactly on the ray in x,y.
    hit = ray_surface_intersect_ex(Point3D(2.3, 1.7, -5), Point3D(0, 0, 1), surf)
    assert hit is not None
    p, _t, hu, hv = hit
    assert abs(p.x - 2.3) < 1e-9 and abs(p.y - 1.7) < 1e-9
    assert (surf.evaluate(hu, hv) - p).length() < 1e-9
    # Section extraction: the cap plane z=3 yields ONE closed loop, every point
    # of which lies on the plane and on the surface (parametric reconnection).
    plane = TrimPlane(Point3D(0, 0, 1), 3.0)
    loops = surface_plane_section(surf, plane)
    closed = [pts for c, pts in loops if c]
    assert len(closed) == 1, f"expected one closed section loop, got {loops}"
    for su, sv, sp in closed[0]:
        assert abs(plane.signed_distance(sp)) < 1e-8
        assert (surf.evaluate(su, sv) - sp).length() < 1e-9
    print(f"  [ok] projection exact, ray contact on true surface, "
          f"closed section loop of {len(closed[0])} pts")


def test_trim_cap_section_ring():
    print("test_trim_cap_section_ring")
    from brep.geometry import NURBSSurface
    # The interior cap cut must UPDATE THE TOPOLOGY: the section curve becomes a
    # real loop of vertices/edges (cap face + inner ring), traversable through
    # the half-edge structure -- not just a render-time tag.
    k = Kernel()
    d = macro.create_nurbs_dome(k, 20.0, 8.0)
    macro.trim_solid_by_plane(k, d, 0, 0, 1, 3.0)
    _assert_valid(d, "dome after cap trim with section ring")
    rep = check_solid(d)
    assert rep.rings == 1, f"section must appear as an inner ring, rings={rep.rings}"
    assert rep.f == 3, f"cap face must be split off, F={rep.f}"
    # The kept cap face's outer loop IS the section: walk it via topology.
    cap = next(f for f in d.faces
               if not getattr(f, "discarded", False)
               and isinstance(f.surface, NURBSSurface))
    ring_verts = [he.vertex for he in cap.outer.halfedges()]
    assert len(ring_verts) >= 12
    for v in ring_verts:
        assert abs(v.point.z - 3.0) < 1e-8, "section vertex must lie on the plane"
        uv = getattr(v, "on_surface_uv", None)
        assert uv is not None, "section vertex must carry its surface parameters"
        assert (cap.surface.evaluate(*uv) - v.point).length() < 1e-9, \
            "section vertex must lie on the surface at its stored (u,v)"
    # keep below: the annulus (outer boundary + ring) survives instead.
    k2 = Kernel()
    d2 = macro.create_nurbs_dome(k2, 20.0, 8.0)
    macro.trim_solid_by_plane(k2, d2, 0, 0, 1, 3.0, keep_below=True)
    _assert_valid(d2, "dome cap trim keep-below")
    keepers = [f for f in d2.faces if not getattr(f, "discarded", False)]
    assert any(len(f.loops) > 1 for f in keepers), \
        "keep-below must retain the ringed annulus face"
    print(f"  [ok] section ring of {len(ring_verts)} verts traversable via topology")


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
    assert (new_v.point - Point3D(0, 0, 4)).length() < 1e-6, \
        f"extended vertex must land on the dome apex, got {new_v.point}"
    # The contact carries its parametric address on the target surface, and the
    # surface evaluated there reproduces the contact point (Newton-refined).
    uv = getattr(new_v, "on_surface_uv", None)
    assert uv is not None, "surface contact must store (u,v) on the target"
    assert (surf.evaluate(*uv) - new_v.point).length() < 1e-9
    print(f"  [ok] wire extended onto NURBS surface at {new_v.point}, uv={uv}")


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


def _crossing_pair(k):
    """Dome A (apex z=4) and dome B flipped into a bowl at z=5 (dip z=1)."""
    from brep.geometry import NURBSSurface
    da = macro.create_nurbs_dome(k, 20.0, 8.0)
    fa = next(f for f in da.faces if isinstance(f.surface, NURBSSurface))
    db = macro.create_nurbs_dome(k, 20.0, 8.0)
    fb = next(f for f in db.faces if isinstance(f.surface, NURBSSurface))
    net = [[Point3D(p.x, p.y, 5.0 - p.z) for p in row]
           for row in fb.surface.control_net]
    fb.surface = NURBSSurface(net, 2, 2)
    for v in db.vertices:
        v.point = Point3D(v.point.x, v.point.y, 5.0)
    return fa, fb


def test_kev_inverse_of_mev():
    print("test_kev_inverse_of_mev")
    k = Kernel()
    s, _f, v0 = eu.mvfs(k, Point3D(0, 0, 0))
    e1, _v1 = eu.mev(k, v0, Point3D(5, 0, 0))
    e2, _v2 = eu.mev(k, _v1, Point3D(5, 5, 0))
    eu.kev(k, e2)
    assert (s.num_vertices, s.num_edges) == (2, 1)
    _assert_valid(s, "wire after KEV of the tip spike")
    eu.kev(k, e1)                       # degenerates back to the MVFS seed
    assert (s.num_vertices, s.num_edges) == (1, 0)
    _e3, _v3 = eu.mev(k, v0, Point3D(1, 1, 0))   # seed must be regrowable
    _assert_valid(s, "wire regrown from restored seed")
    # KEV must refuse a manifold (non-spike) edge.
    k2 = Kernel()
    b = macro.create_box(k2, 5, 5, 5)
    try:
        eu.kev(k2, b.edges[0])
        assert False, "KEV accepted a manifold edge"
    except ValueError:
        pass
    print("  [ok] KEV removes spikes, restores the seed, rejects manifold edges")


def test_surface_surface_intersection():
    print("test_surface_surface_intersection")
    k = Kernel()
    fa, fb = _crossing_pair(k)
    wire, closed, n = macro.intersect_surfaces(k, fa, fb)
    assert closed and n >= 16
    _assert_valid(wire, "SSI wire")
    # Every wire vertex lies on BOTH surfaces at its stored parameters.
    ra = max((fa.surface.evaluate(*v.on_surface_uv) - v.point).length()
             for v in wire.vertices)
    rb = max((fb.surface.evaluate(*v.on_surface_uv_b) - v.point).length()
             for v in wire.vertices)
    assert ra < 1e-6 and rb < 1e-6, f"SSI residuals too large: {ra}, {rb}"
    print(f"  [ok] closed SSI loop of {n} pts, on both surfaces "
          f"(residuals {ra:.1e}/{rb:.1e})")


def test_trim_by_surface():
    print("test_trim_by_surface")
    from brep.geometry import NURBSSurface, SurfaceCutter
    k = Kernel()
    dome = macro.create_nurbs_dome(k, 20.0, 8.0)
    fa = next(f for f in dome.faces if isinstance(f.surface, NURBSSurface))
    box = macro.create_box(k, 8, 8, 8, origin=Point3D(-4, -4, 0))
    res = macro.trim_solid_by_surface(k, box, fa, keep_below=True)
    assert res.is_topological and res.n_cut > 0 and res.n_discard > 0
    _assert_valid(box, "box trimmed by a curved NURBS cutter")
    # Split vertices must land ON the curved cutter, not on a linear chord.
    cutter = SurfaceCutter(fa.surface)
    split_vs = [v for v in box.vertices
                if 0.01 < abs(v.point.z) and abs(v.point.z - 8) > 0.01
                and abs(v.point.z) > 0.01 and 0.01 < v.point.z < 7.99]
    assert split_vs, "expected split vertices on the box sides"
    worst = max(abs(cutter.signed_distance(v.point)) for v in split_vs)
    assert worst < 1e-6, f"split vertex off the cutter surface by {worst}"
    # The section must FOLLOW the curved intersection, not jump across it with
    # one straight chord per face: every section sub-edge midpoint stays close
    # to the cutter (a single chord would deviate by ~0.32 here).
    sub_edges = [e for e in box.edges
                 if 0.01 < e.he1.vertex.point.z < 7.99
                 and 0.01 < e.he1.end_vertex.point.z < 7.99]
    assert len(sub_edges) >= 16, "cut edges must be refined into a polyline"
    sag = max(abs(cutter.signed_distance(
        (e.he1.vertex.point + e.he1.end_vertex.point) * 0.5))
        for e in sub_edges)
    assert sag < 0.05, f"section polyline strays {sag} from the true curve"
    print(f"  [ok] box carved by dome: cut={res.n_cut}, "
          f"{len(split_vs)} split verts on the cutter (max {worst:.1e}), "
          f"section polyline sagitta {sag:.3f}")


def test_blend_patch():
    print("test_blend_patch")
    from brep.geometry import (NURBSSurface, surface_closest_point,
                               surface_normal)
    k = Kernel()
    fa, fb = _crossing_pair(k)
    bl = macro.blend_surfaces(k, fa, fb, width=1.5, samples=9)
    _assert_valid(bl, "blend patch lamina")
    patch = next(f.surface for f in bl.faces if f.surface)
    assert patch.degree_v == 5, "blend cross-sections must be quintic"
    m = patch.n_u
    # The patch boundary must INTERPOLATE the rails: at every sample station
    # the v=0 / v=1 edges lie on surface A / B respectively.
    for kk in range(m):
        t = kk / (m - 1)
        pa = patch.evaluate(t, 0.0)
        _u, _v, foot_a = surface_closest_point(fa.surface, pa)
        assert (foot_a - pa).length() < 1e-6, "rail A station off surface A"
        pb = patch.evaluate(t, 1.0)
        _u, _v, foot_b = surface_closest_point(fb.surface, pb)
        assert (foot_b - pb).length() < 1e-6, "rail B station off surface B"
    # Tangent continuity: the cross-boundary tangent at a station lies in
    # surface A's tangent plane (dot with the normal ~ 0).
    h = 1e-4
    p0 = patch.evaluate(0.5, 0.0)
    p1 = patch.evaluate(0.5, h)
    d1 = (p1 - p0) * (1.0 / h)
    u, v, _foot = surface_closest_point(fa.surface, p0)
    n_dot = abs(d1.normalized().dot(surface_normal(fa.surface, u, v)))
    assert n_dot < 5e-3, f"cross tangent leaves the tangent plane: {n_dot}"
    print(f"  [ok] quintic blend interpolates both rails at {m} stations, "
          f"tangent-plane deviation {n_dot:.1e}")


def test_pcurve_step_export():
    print("test_pcurve_step_export")
    import io as _io
    from brep import stepio
    k = Kernel()
    d = macro.create_nurbs_dome(k, 20.0, 8.0)
    macro.trim_solid_by_plane(k, d, 0, 0, 1, 3.0)      # cap ring trim
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "cap.step")
        stepio.save(d, fp)
        text = open(fp).read()
        # Analytic: ONE trimmed ADVANCED_FACE over the full B-spline surface,
        # every section edge carrying a PCURVE in (u,v) space.
        assert text.count("ADVANCED_FACE") == 1
        assert text.count("B_SPLINE_SURFACE_WITH_KNOTS") == 1
        assert text.count("PCURVE(") == 24
        assert text.count("DEFINITIONAL_REPRESENTATION") == 24
        assert "OPEN_SHELL" in text
        fp2 = os.path.join(td, "cap_faceted.step")
        stepio.save(d, fp2, faceted=True)
        t2 = open(fp2).read()
        assert t2.count("ADVANCED_FACE") > 100 and "PCURVE(" not in t2
    print("  [ok] trimmed NURBS exports analytically (surface + 24 pcurves); "
          "faceted fallback intact")


def test_transform_updates_geometry():
    print("test_transform_updates_geometry")
    from brep.controller import BRepShell
    from brep.geometry import NURBSSurface
    import io as _io, contextlib
    sh = BRepShell()
    with contextlib.redirect_stdout(_io.StringIO()):
        sh.onecmd("create nurbs 20 8 as @d")
        sh.onecmd("rotate @d x 180")
        sh.onecmd("move @d 0 0 5")
    solid = sh.kernel.solids[0]
    surf = next(f.surface for f in solid.faces if f.surface)
    apex = surf.evaluate(0.5, 0.5)
    # Rotated 180 deg about x then raised by 5: the apex z=4 must land at z=1.
    assert abs(apex.z - 1.0) < 1e-9, f"surface did not follow the transform: {apex}"
    # And the topology moved with it (lamina verts at z=5).
    assert all(abs(v.point.z - 5.0) < 1e-9 for v in solid.vertices)
    print("  [ok] move/rotate transform BOTH the vertices and the NURBS net")


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
        test_numeric_intersection,
        test_trim_cap_section_ring,
        test_extend_curve_to_plane,
        test_extend_curve_to_surface,
        test_extend_face_to_plane,
        test_extend_face_to_surface,
        test_kev_inverse_of_mev,
        test_surface_surface_intersection,
        test_trim_by_surface,
        test_blend_patch,
        test_pcurve_step_export,
        test_transform_updates_geometry,
        test_kef_inverse_of_mef,
    ]
    for t in tests:
        t()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_all()
