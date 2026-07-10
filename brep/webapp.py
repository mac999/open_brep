"""
The authoring web app - a browser front-end for the same kernel the REPL drives.

``brep> webapp`` starts a small stdlib HTTP server on localhost, bound to the
*live* :class:`~brep.controller.BRepShell`. There is one Kernel, one id registry,
one alias table: a box created in the browser shows up in ``list`` at the prompt,
and ``trim @b by plane ...`` typed at the prompt is visible in the canvas on the
next refresh. That shared state is the point of the tool - it makes B-Rep
operations something you can see while you test them.

Layout of the API (all JSON, all POST unless noted):
    GET  /api/scene           tessellated meshes + stats for every solid
    GET  /api/entities        surviving face/edge/vertex ids per solid
                              (fills the id pickers of the modeling-op menus)
    GET  /api/entity?oid=N    geometric + topological detail of one
                              face/edge/vertex (the right-hand props panel)
    POST /api/create          {kind, params}      -> build a primitive
    POST /api/delete          {oid}
    POST /api/transform       {oid, op, ...}      -> move / rotate / scale
    POST /api/rename          {oid, name}
    POST /api/save            {path, oids, faceted}
    POST /api/load            {path}
    GET  /api/files           STEP files near the working directory
    POST /api/command         {line}              -> run any CLI command

Every mutation takes ``_LOCK`` so a request cannot interleave with a command
being typed at the REPL.
"""

from __future__ import annotations

import contextlib
import io
import json
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from . import macro, stepio, xform
from .geometry import (Bezier, NURBSSurface, rotation_matrix, scaling_matrix,
                       translation_matrix)
from .topology import Edge, Face, Vertex
from .validate import check_solid
from .viewer import _fan_tris, _nurbs_face_mesh, _surviving_refs

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# One server per process; ``webapp`` twice just re-opens the browser.
_SERVER: Optional["_WebApp"] = None
_LOCK = threading.RLock()

# Surface sampling density for NURBS faces sent to the canvas.
_NURBS_NU = _NURBS_NV = 14


# --------------------------------------------------------------------------- #
# Scene serialization
# --------------------------------------------------------------------------- #
def _solid_mesh(solid):
    """Triangulate every surviving face into one (positions, triangles) buffer.

    Also returns the owning face oid per triangle so the canvas can pick faces.
    """
    positions: List[List[float]] = []
    tris: List[List[int]] = []
    tri_faces: List[int] = []
    for f in solid.faces:
        if getattr(f, "discarded", False) or f.outer is None:
            continue
        if isinstance(f.surface, NURBSSurface):
            pts, face_tris = _nurbs_face_mesh(f, _NURBS_NU, _NURBS_NV)
        else:
            pts = [he.vertex.point for he in f.outer.halfedges()
                   if he.vertex and he.vertex.point]
            if len(pts) < 3:
                continue
            face_tris = _fan_tris(len(pts))
        base = len(positions)
        positions.extend([p.x, p.y, p.z] for p in pts)
        tris.extend([a + base, b + base, c + base] for a, b, c in face_tris)
        tri_faces.extend(f.oid for _ in face_tris)
    return positions, tris, tri_faces


def _solid_wire(solid):
    """Line segments (with edge oids) for every edge referenced by a surviving face."""
    _v_oids, e_oids = _surviving_refs(solid)
    segments = []
    edge_oids = []
    for e in solid.edges:
        if e.oid not in e_oids or e.he1 is None:
            continue
        a, b = e.he1.vertex, e.he1.end_vertex
        if a is None or b is None or a.point is None or b.point is None:
            continue
        segments.append([[a.point.x, a.point.y, a.point.z],
                         [b.point.x, b.point.y, b.point.z]])
        edge_oids.append(e.oid)
    return segments, edge_oids


def _solid_verts(solid):
    """Positioned surviving vertices, for vertex picking/markers in the canvas."""
    v_oids, _e_oids = _surviving_refs(solid)
    return [{"oid": v.oid, "p": [v.point.x, v.point.y, v.point.z]}
            for v in solid.vertices
            if v.oid in v_oids and v.point is not None]


def _face_grips(solid):
    """One on-surface grip point per surviving face (the canvas pick handles).

    A NURBS face's boundary centroid can float off the surface (a dome's four
    base corners average to the base plane), so grips are evaluated on the
    surface at the parametric middle instead.
    """
    grips = []
    for f in solid.faces:
        if getattr(f, "discarded", False) or f.outer is None:
            continue
        if isinstance(f.surface, NURBSSurface):
            p = f.surface.evaluate(0.5, 0.5)   # components are numpy floats
        else:
            p = xform.centroid(xform.vertices_of(f))
        grips.append({"oid": f.oid, "p": [float(p.x), float(p.y), float(p.z)]})
    return grips


def _solid_payload(solid) -> dict:
    positions, tris, tri_faces = _solid_mesh(solid)
    wire, wire_edges = _solid_wire(solid)
    lo, hi = xform.bounding_box(solid)
    centre = xform.centroid(solid.vertices)
    report = check_solid(solid)
    return {
        "oid": solid.oid,
        "name": solid.name,
        "kind": (solid.name or "solid").split(":")[0].split("(")[0] or "solid",
        "positions": positions,
        "triangles": tris,
        "triFaces": tri_faces,
        "wire": wire,
        "wireEdges": wire_edges,
        "verts": _solid_verts(solid),
        "faceGrips": _face_grips(solid),
        "stats": {
            "v": solid.num_vertices, "e": solid.num_edges,
            "f": solid.num_faces, "rings": solid.num_rings,
            "shells": solid.shells, "genus": solid.genus,
            "nurbsFaces": sum(1 for f in solid.faces
                              if isinstance(f.surface, NURBSSurface)),
        },
        "bbox": {"min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z]},
        "centroid": [centre.x, centre.y, centre.z],
        "valid": report.passed,
        "eulerLhs": report.euler_lhs,
        "eulerRhs": report.euler_rhs,
        "pointerErrors": report.pointer_errors[:4],
    }


def _entities_payload(kernel) -> List[dict]:
    """Surviving face/edge/vertex ids per solid, for the web UI's id pickers.

    Reuses the same survivor logic as the mesh/wire serializers so a trimmed
    solid only offers ids the CLI commands can actually operate on.
    """
    out = []
    for solid in kernel.solids:
        v_oids, e_oids = _surviving_refs(solid)
        faces = [{"oid": f.oid,
                  "nurbs": isinstance(f.surface, NURBSSurface)}
                 for f in solid.faces
                 if not getattr(f, "discarded", False) and f.outer is not None]
        out.append({
            "oid": solid.oid,
            "name": solid.name,
            "faces": faces,
            "edges": sorted(e_oids),
            "vertices": sorted(v_oids),
        })
    return out


def _face_mesh_one(face):
    """(points, triangles) of a single face, same sampling as the scene mesh."""
    if isinstance(face.surface, NURBSSurface):
        return _nurbs_face_mesh(face, _NURBS_NU, _NURBS_NV)
    pts = [he.vertex.point for he in face.outer.halfedges()
           if he.vertex and he.vertex.point]
    return pts, (_fan_tris(len(pts)) if len(pts) >= 3 else [])


def _entity_detail(kernel, oid: int) -> dict:
    """Geometric + topological properties of one sub-entity, for the props panel."""
    entity = kernel.get(oid)
    solid = kernel.solid_of(entity)
    out = {"oid": oid, "solid": solid.oid if solid else None,
           "solidName": solid.name if solid else None}

    if isinstance(entity, Vertex):
        p = entity.point
        out.update({"type": "vertex",
                    "point": [p.x, p.y, p.z] if p else None})
        return out

    if isinstance(entity, Edge):
        a = entity.he1.vertex if entity.he1 else None
        b = entity.he1.end_vertex if entity.he1 else None
        curve = getattr(entity, "curve", None)
        detail = {"type": "edge",
                  "curve": (f"bezier (deg {curve.degree})"
                            if isinstance(curve, Bezier) else "line"),
                  "a": None, "b": None, "aOid": a.oid if a else None,
                  "bOid": b.oid if b else None, "length": None, "centroid": None}
        if a and b and a.point and b.point:
            pa, pb = a.point, b.point
            detail["a"] = [pa.x, pa.y, pa.z]
            detail["b"] = [pb.x, pb.y, pb.z]
            detail["length"] = ((pb.x - pa.x) ** 2 + (pb.y - pa.y) ** 2
                                + (pb.z - pa.z) ** 2) ** 0.5
            detail["centroid"] = [(pa.x + pb.x) / 2, (pa.y + pb.y) / 2,
                                  (pa.z + pb.z) / 2]
        out.update(detail)
        return out

    if isinstance(entity, Face):
        # A trim leaves discarded faces in solid.faces; they have no boundary
        # to mesh, so refuse them instead of crashing on face.outer.
        if getattr(entity, "discarded", False) or entity.outer is None:
            raise ValueError(f"face #{oid} has no surviving boundary")
        centre = xform.centroid(xform.vertices_of(entity))
        pts, tris = _face_mesh_one(entity)
        area = 0.0
        normal = [0.0, 0.0, 0.0]
        for ia, ib, ic in tris:
            pa, pb, pc = pts[ia], pts[ib], pts[ic]
            u = (pb.x - pa.x, pb.y - pa.y, pb.z - pa.z)
            v = (pc.x - pa.x, pc.y - pa.y, pc.z - pa.z)
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0])
            area += 0.5 * (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
            normal = [normal[i] + n[i] for i in range(3)]
        n_len = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
        surf = entity.surface
        out.update({
            "type": "face",
            "surface": (f"nurbs (deg {surf.degree_u}x{surf.degree_v})"
                        if isinstance(surf, NURBSSurface) else "plane"),
            "loops": len(entity.loops),
            "numVertices": len(xform.vertices_of(entity)),
            "centroid": [centre.x, centre.y, centre.z],
            "area": area,
            "normal": ([normal[i] / n_len for i in range(3)]
                       if n_len > 1e-12 else None),
        })
        return out

    raise ValueError(f"#{oid} is not a vertex, edge or face")


# --------------------------------------------------------------------------- #
# Modeling operations (the API's verbs)
# --------------------------------------------------------------------------- #
_PRIMITIVES = ("box", "sphere", "cylinder", "plane", "nurbs")


def _create(kernel, kind: str, params: dict):
    def num(key, default, cast=float):
        try:
            return cast(params.get(key, default))
        except (TypeError, ValueError):
            return cast(default)

    if kind == "box":
        return macro.create_box(kernel, num("length", 10), num("width", 10),
                                num("height", 10))
    if kind == "sphere":
        return macro.create_sphere(kernel, num("radius", 5),
                                   max(3, num("slices", 16, int)),
                                   max(2, num("stacks", 8, int)))
    if kind == "cylinder":
        return macro.create_cylinder(kernel, num("radius", 4), num("height", 10),
                                     max(3, num("slices", 16, int)))
    if kind == "plane":
        return macro.create_plane(kernel, num("width", 20), num("height", 20))
    if kind == "nurbs":
        return macro.create_nurbs_dome(kernel, num("size", 20), num("height", 8))
    raise ValueError(f"unknown primitive '{kind}'")


def _transform(kernel, solid, body: dict) -> str:
    op = (body.get("op") or "").lower()
    if op == "move":
        d = [float(body.get(k, 0.0)) for k in ("dx", "dy", "dz")]
        xform.apply_transform(solid, translation_matrix(*d))
        return f"moved #{solid.oid} by ({d[0]}, {d[1]}, {d[2]})"
    if op == "rotate":
        axis = (body.get("axis") or "z").lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be x, y or z")
        angle = float(body.get("angle", 0.0))
        # Rotate about the solid's own centroid: an in-place spin is what the
        # gizmo implies, whereas the CLI's 'rotate' turns about the origin.
        centre = xform.centroid(solid.vertices)
        xform.apply_transform(solid, translation_matrix(-centre.x, -centre.y, -centre.z))
        xform.apply_transform(solid, rotation_matrix(axis, angle))
        xform.apply_transform(solid, translation_matrix(centre.x, centre.y, centre.z))
        return f"rotated #{solid.oid} about {axis} by {angle} deg (about its centroid)"
    if op == "scale":
        factor = float(body.get("factor", 1.0))
        if abs(factor) < 1e-9:
            raise ValueError("scale factor must be non-zero")
        centre = xform.centroid(solid.vertices)
        xform.apply_transform(solid, scaling_matrix(factor, centre))
        return f"scaled #{solid.oid} by {factor}"
    raise ValueError(f"unknown transform '{op}' (move, rotate, scale)")


def _list_step_files(root: str) -> List[str]:
    """STEP files in ``root`` and its immediate sub-directories, newest first."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if os.path.relpath(dirpath, root) == ".":
            depth = -1
        if depth >= 1:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
        for fn in filenames:
            if fn.lower().endswith((".step", ".stp")):
                path = os.path.join(dirpath, fn)
                found.append((os.path.getmtime(path),
                              os.path.relpath(path, root).replace("\\", "/")))
    found.sort(reverse=True)
    return [p for _mtime, p in found[:200]]


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    server_version = "brep-webapp"

    # The shell is attached to the server object by _WebApp.
    @property
    def shell(self):
        return self.server.shell            # type: ignore[attr-defined]

    @property
    def kernel(self):
        return self.server.shell.kernel     # type: ignore[attr-defined]

    def log_message(self, fmt, *args) -> None:
        pass  # keep the REPL prompt clean

    # -- responses ------------------------------------------------------- #
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- routes ---------------------------------------------------------- #
    def do_GET(self) -> None:
        route = urlparse(self.path)
        path = route.path
        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/scene":
            with _LOCK:
                solids = [_solid_payload(s) for s in self.kernel.solids]
            return self._json({"ok": True, "solids": solids,
                               "cwd": os.getcwd().replace("\\", "/")})
        if path == "/api/entities":
            with _LOCK:
                solids = _entities_payload(self.kernel)
            return self._json({"ok": True, "solids": solids})
        if path == "/api/entity":
            try:
                oid = int(parse_qs(route.query).get("oid", [""])[0])
                with _LOCK:
                    detail = _entity_detail(self.kernel, oid)
            except (ValueError, KeyError, TypeError) as exc:
                return self._error(str(exc))
            return self._json({"ok": True, "entity": detail})
        if path == "/api/files":
            root = parse_qs(route.query).get("dir", ["."])[0]
            try:
                return self._json({"ok": True, "files": _list_step_files(root)})
            except OSError as exc:
                return self._error(str(exc))
        return self._error("not found", 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._error(f"malformed request body: {exc}")

        handlers = {
            "/api/create": self._post_create,
            "/api/delete": self._post_delete,
            "/api/transform": self._post_transform,
            "/api/rename": self._post_rename,
            "/api/save": self._post_save,
            "/api/load": self._post_load,
            "/api/command": self._post_command,
        }
        handler = handlers.get(path)
        if handler is None:
            return self._error("not found", 404)
        try:
            with _LOCK:
                handler(body)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self._error(str(exc))

    def _post_create(self, body: dict) -> None:
        kind = (body.get("kind") or "").lower()
        if kind not in _PRIMITIVES:
            raise ValueError(f"unknown primitive '{kind}'")
        solid = _create(self.kernel, kind, body.get("params") or {})
        self._json({"ok": True, "oid": solid.oid,
                    "message": f"+ created Solid #{solid.oid} ({kind})"})

    def _post_delete(self, body: dict) -> None:
        solid = self._solid(body.get("oid"))
        oid = solid.oid
        self.kernel.delete_solid(solid)
        dead = [a for a, target in self.shell.aliases.items()
                if self.kernel.find(target) is None]
        for a in dead:
            del self.shell.aliases[a]
        self._json({"ok": True, "message": f"- deleted Solid #{oid}"})

    def _post_transform(self, body: dict) -> None:
        solid = self._solid(body.get("oid"))
        self._json({"ok": True, "message": _transform(self.kernel, solid, body)})

    def _post_rename(self, body: dict) -> None:
        solid = self._solid(body.get("oid"))
        name = str(body.get("name") or "").strip()
        old = solid.name
        solid.name = name
        self._json({"ok": True,
                    "message": f"renamed #{solid.oid} '{old}' -> '{name}'"})

    def _post_save(self, body: dict) -> None:
        path = (body.get("path") or "").strip()
        if not path:
            raise ValueError("no output path given")
        oids = body.get("oids")
        if oids:
            solids = [self._solid(o) for o in oids]
        else:
            solids = list(self.kernel.solids)
        if not solids:
            raise ValueError("no solids to save")
        stepio.save_solids(solids, path, faceted=bool(body.get("faceted")))
        names = ", ".join(f"#{s.oid}" for s in solids)
        self._json({"ok": True,
                    "message": f"saved {len(solids)} solid(s) [{names}] -> {path}"})

    def _post_load(self, body: dict) -> None:
        path = (body.get("path") or "").strip()
        if not path:
            raise ValueError("no input path given")
        if not os.path.exists(path):
            raise ValueError(f"no such file: {path}")
        if body.get("replace"):
            for s in list(self.kernel.solids):
                self.kernel.delete_solid(s)
            self.shell.aliases.clear()
        solids = stepio.load(self.kernel, path)
        summary = ", ".join(
            f"#{s.oid} '{s.name}' V={s.num_vertices} E={s.num_edges} F={s.num_faces}"
            for s in solids)
        self._json({"ok": True, "oids": [s.oid for s in solids],
                    "message": f"loaded {len(solids)} solid(s) from {path}: {summary}"})

    def _post_command(self, body: dict) -> None:
        line = (body.get("line") or "").strip()
        if not line:
            raise ValueError("empty command")
        head = line.split()[0].lower()
        if head in ("exit", "quit", "eof", "webapp", "view"):
            raise ValueError(f"'{head}' is not available from the web console")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.shell.onecmd(line)
        self._json({"ok": True, "output": buffer.getvalue().rstrip("\n")})

    def _solid(self, oid):
        if oid is None:
            raise ValueError("no solid id given")
        entity = self.kernel.get(int(oid))
        solid = self.kernel.solid_of(entity)
        if solid is None:
            raise ValueError(f"#{oid} is not part of any solid")
        return solid

    # -- static files ----------------------------------------------------- #
    def _serve_static(self, relative: str) -> None:
        target = os.path.normpath(os.path.join(WEB_DIR, relative))
        if not target.startswith(WEB_DIR) or not os.path.isfile(target):
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        with open(target, "rb") as fh:
            self._send(200, fh.read(), ctype)


class _WebApp:
    """A background HTTP server bound to a live shell."""

    def __init__(self, shell, host: str, port: int):
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.shell = shell          # type: ignore[attr-defined]
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[:2]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       name="brep-webapp", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def start(shell, port: int = 8765, host: str = "127.0.0.1",
          open_browser: bool = True) -> str:
    """
    Start (or re-use) the authoring web app for ``shell`` and return its URL.

    Bound to the loopback interface only: the app executes kernel commands, so
    it must not be reachable from the network.
    """
    global _SERVER
    if _SERVER is None:
        try:
            _SERVER = _WebApp(shell, host, port)
        except OSError:
            _SERVER = _WebApp(shell, host, 0)   # requested port busy; let the OS pick
        _SERVER.start()
    if open_browser:
        webbrowser.open(_SERVER.url)
    return _SERVER.url


def stop() -> None:
    """Shut the server down (used by the tests)."""
    global _SERVER
    if _SERVER is not None:
        _SERVER.stop()
        _SERVER = None


def is_running() -> bool:
    return _SERVER is not None


def url() -> Optional[str]:
    return _SERVER.url if _SERVER else None
