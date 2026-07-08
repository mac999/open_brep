"""
Controller - the REPL and batch test runner (the 'C' in MVC).

Parses CLI input with ``cmd``/``shlex``, routes micro/macro commands to the
kernel, and renders results through the View layer. It owns no topology logic of
its own: every mutation goes through euler_ops / macro, every read through view.
"""

from __future__ import annotations

import cmd
import shlex
import sys
from typing import List, Optional

from . import euler_ops as eu
from . import macro
from . import stepio
from . import view
from .geometry import (
    Bezier,
    NURBSSurface,
    Point3D,
    TrimPlane,
    apply_matrix,
    rotation_matrix,
    scaling_matrix,
    translation_matrix,
)
from .model import Kernel
from .topology import Edge, Face, Loop, Solid, Vertex
from .validate import check_solid


class CliError(Exception):
    """Raised for user-facing command errors (kept distinct from real bugs)."""


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise CliError(message)


class BRepShell(cmd.Cmd):
    INTRO = (
        "B-Rep CLI Kernel  -  type 'help' for commands, 'exit' to quit.\n"
        "Try:  create box 10 10 10 as @b   then   check validity @b\n"
        "Refer to entities by #id, by @alias ('as @name' / 'set'), or by the\n"
        "last created of a kind: $solid $vertex $edge $face $loop $last ('vars').\n"
    )
    PROMPT = "brep> "
    # cmd's own intro/prompt are silenced: we print them ourselves with an
    # explicit flush. The prompt has no trailing newline, so on block-buffered
    # streams (e.g. Python under Git Bash/mintty, where stdout is not a TTY)
    # cmd's input(self.prompt) leaves it stuck in the buffer until the next I/O.
    # Flushing it ourselves makes the prompt appear immediately.
    intro = ""
    prompt = ""

    # Symbolic references understood anywhere a #id is expected.
    LAST_KINDS = ("solid", "vertex", "edge", "face", "loop", "halfedge", "last")

    def __init__(self, kernel: Optional[Kernel] = None):
        super().__init__()
        self.kernel = kernel or Kernel()
        self.aliases: dict[str, int] = {}  # @name -> oid

    # ------------------------------------------------------------------ #
    # Prompt handling (flush so it shows on block-buffered terminals)
    # ------------------------------------------------------------------ #
    def preloop(self) -> None:
        # cmdloop() runs this before its (silenced) intro, so we print both the
        # banner and the first prompt here, flushed.
        sys.stdout.write(self.INTRO)
        self._emit_prompt()

    def postcmd(self, stop: bool, line: str) -> bool:
        if not stop:
            self._emit_prompt()  # next prompt, flushed
        return stop

    def _emit_prompt(self) -> None:
        sys.stdout.write(self.PROMPT)
        sys.stdout.flush()

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def _out(self, text: str) -> None:
        print(text)
        sys.stdout.flush()

    def _resolve_id(self, token: str) -> int:
        """
        Resolve an id token to an integer oid. Accepts three forms:
            #123 / 123   literal id
            @name        a user alias bound via 'as @name' or 'set'
            $vertex ...  the most recently created entity of that kind
                         ($solid/$vertex/$edge/$face/$loop/$last)
        """
        token = token.strip()
        if token.startswith("@"):
            name = token[1:]
            if name not in self.aliases:
                raise CliError(f"unknown alias @{name} (see 'vars')")
            return self.aliases[name]
        if token.startswith("$"):
            kind = token[1:].lower()
            if kind not in self.LAST_KINDS:
                raise CliError(f"unknown $-variable ${kind}; "
                               f"use one of {', '.join('$' + k for k in self.LAST_KINDS)}")
            entity = self.kernel.registry.last(kind)
            if entity is None:
                raise CliError(f"no {kind} has been created yet for ${kind}")
            return entity.oid
        if token.startswith("#"):
            token = token[1:]
        if not token.lstrip("-").isdigit():
            raise CliError(f"'{token}' is not a valid #id, @alias or $var")
        return int(token)

    def _bind_alias(self, as_spec, entity) -> None:
        """If the 'as' clause requested an @alias, bind it to the created entity."""
        if as_spec and as_spec[0] == "alias":
            self.aliases[as_spec[1]] = entity.oid

    def _resolve_solid(self, arg_tokens: List[str]) -> Optional[Solid]:
        """Resolve an optional trailing '#id' into a solid (or None for 'all')."""
        if not arg_tokens:
            return None
        entity = self.kernel.get(self._resolve_id(arg_tokens[0]))
        solid = self.kernel.solid_of(entity)
        _expect(solid is not None, f"#{entity.oid} is not part of any solid")
        return solid

    def onecmd(self, line: str):
        """Strip comments, then dispatch so command errors don't crash the REPL."""
        line = self._strip_comment(line)
        if not line.strip():
            return False
        try:
            return super().onecmd(line)
        except CliError as exc:
            self._out(f"ERROR: {exc}")
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            self._out(f"ERROR: {exc}")
        return False

    @staticmethod
    def _strip_comment(line: str) -> str:
        """
        Remove comments. A line whose first token is '#' or '//' is a full
        comment; '//' anywhere starts an inline comment. '#' is NOT treated as an
        inline marker because it collides with #ids (e.g. 'as #102').
        """
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            return ""
        idx = line.find("//")
        return line[:idx] if idx != -1 else line

    def emptyline(self) -> bool:  # don't repeat last command on blank input
        return False

    # ------------------------------------------------------------------ #
    # 4.1  Micro topology commands
    # ------------------------------------------------------------------ #
    def do_micro(self, arg: str) -> None:
        "micro mvfs|mev|mef|kev|kef|semv|kemr|mekr|kfmr|mfkr ...  - atomic Euler operators"
        tokens = shlex.split(arg)
        _expect(bool(tokens),
                "usage: micro <mvfs|mev|mef|kev|kef|semv|kemr|mekr|kfmr|mfkr> ...")
        op = tokens[0].lower()
        rest = tokens[1:]
        if op == "mvfs":
            self._micro_mvfs(rest)
        elif op == "mev":
            self._micro_mev(rest)
        elif op == "mef":
            self._micro_mef(rest)
        elif op == "kev":
            self._micro_kev(rest)
        elif op == "kef":
            self._micro_kef(rest)
        elif op in ("semv", "split"):
            self._micro_semv(rest)
        elif op == "kemr":
            self._micro_kemr(rest)
        elif op == "mekr":
            self._micro_mekr(rest)
        elif op == "kfmr":
            self._micro_kfmr(rest)
        elif op == "mfkr":
            self._micro_mfkr(rest)
        else:
            raise CliError(f"unknown micro operator '{op}'")

    def _split_as(self, tokens: List[str]) -> tuple[List[str], Optional[tuple]]:
        """
        Strip a trailing 'as ...' clause. Returns (remaining_tokens, as_spec)
        where as_spec is None, ('id', int) for 'as #id', or ('alias', name) for
        'as @name'.
        """
        if len(tokens) >= 2 and tokens[-2].lower() == "as":
            target = tokens[-1]
            if target.startswith("@"):
                return tokens[:-2], ("alias", target[1:])
            return tokens[:-2], ("id", self._resolve_id(target))
        return tokens, None

    @staticmethod
    def _explicit_id(as_spec) -> Optional[int]:
        """The explicit oid requested by an 'as #id' clause, else None."""
        return as_spec[1] if as_spec and as_spec[0] == "id" else None

    def _micro_mvfs(self, tokens: List[str]) -> None:
        tokens, as_spec = self._split_as(tokens)
        _expect(len(tokens) == 3, "usage: micro mvfs <x> <y> <z> [as #id|@name]")
        x, y, z = (float(t) for t in tokens)
        solid, face, vertex = eu.mvfs(
            self.kernel, Point3D(x, y, z), solid_oid=self._explicit_id(as_spec))
        self._bind_alias(as_spec, solid)
        self._out(view.format_entity_created("Solid", solid.oid,
                  f"(vertex #{vertex.oid}, face #{face.oid})"))

    def _micro_mev(self, tokens: List[str]) -> None:
        tokens, as_spec = self._split_as(tokens)
        _expect(len(tokens) == 4, "usage: micro mev #<vertex> <x> <y> <z> [as #edge|@name]")
        v = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(v, Vertex), f"{tokens[0]} is not a vertex")
        x, y, z = (float(t) for t in tokens[1:])
        edge, v2 = eu.mev(self.kernel, v, Point3D(x, y, z),
                          edge_oid=self._explicit_id(as_spec))
        self._bind_alias(as_spec, edge)
        self._out(view.format_entity_created("Edge", edge.oid,
                  f"(new vertex #{v2.oid})"))

    def _micro_mef(self, tokens: List[str]) -> None:
        tokens, as_spec = self._split_as(tokens)
        _expect(len(tokens) == 2, "usage: micro mef #<v1> #<v2> [as #face|@name]")
        v1 = self.kernel.get(self._resolve_id(tokens[0]))
        v2 = self.kernel.get(self._resolve_id(tokens[1]))
        _expect(isinstance(v1, Vertex) and isinstance(v2, Vertex),
                "both arguments to mef must be vertices")
        edge, face = eu.mef(self.kernel, v1, v2)
        oid = self._explicit_id(as_spec)
        if oid is not None:  # honour explicit face id by re-registering
            self.kernel.registry.unregister(face.oid)
            self.kernel.registry.register(face, oid)
        self._bind_alias(as_spec, face)
        self._out(view.format_entity_created("Face", face.oid,
                  f"(edge #{edge.oid})"))

    def _micro_kev(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 1, "usage: micro kev #<edge>")
        edge = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(edge, Edge), f"{tokens[0]} is not an edge")
        eu.kev(self.kernel, edge)
        self._out(f"KEV: killed spike edge #{edge.oid} and its tip vertex (-1V -1E)")

    def _micro_kef(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 1, "usage: micro kef #<edge>")
        edge = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(edge, Edge), f"{tokens[0]} is not an edge")
        eu.kef(self.kernel, edge)
        self._out(f"KEF: killed edge #{edge.oid}, merged its two faces (-1E -1F)")

    def _micro_semv(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 4, "usage: micro semv #<edge> <x> <y> <z>")
        edge = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(edge, Edge), f"{tokens[0]} is not an edge")
        x, y, z = (float(t) for t in tokens[1:])
        new_edge, v = eu.split_edge(self.kernel, edge, Point3D(x, y, z))
        self._out(f"SEMV: split edge #{edge.oid} -> new vertex #{v.oid}, "
                  f"new edge #{new_edge.oid} (+1V +1E)")

    def _micro_kemr(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 1, "usage: micro kemr #<edge>")
        edge = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(edge, Edge), f"{tokens[0]} is not an edge")
        ring, _dead = eu.kemr(self.kernel, edge.he1)
        self._out(f"KEMR: killed edge #{edge.oid}, made ring loop #{ring.oid} "
                  f"(-1E +1R)")

    def _micro_mekr(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 2, "usage: micro mekr #<v_outer> #<v_ring>")
        v_out = self.kernel.get(self._resolve_id(tokens[0]))
        v_ring = self.kernel.get(self._resolve_id(tokens[1]))
        _expect(isinstance(v_out, Vertex) and isinstance(v_ring, Vertex),
                "both arguments to mekr must be vertices")
        he_outer = he_ring = None
        for lp in eu._loops_around(v_out):
            face = lp.face
            for ring in face.inner:
                cand = eu.find_outgoing_in_loop(ring, v_ring)
                if cand is not None:
                    he_outer = eu.find_outgoing_in_loop(face.outer, v_out)
                    he_ring = cand
                    break
            if he_ring is not None:
                break
        _expect(he_outer is not None and he_ring is not None,
                "mekr: the two vertices must lie on the outer loop and an "
                "inner ring of the same face")
        edge = eu.mekr(self.kernel, he_outer, he_ring)
        self._out(f"MEKR: bridged ring to outer loop with edge #{edge.oid} "
                  f"(+1E -1R)")

    def _micro_kfmr(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 1, "usage: micro kfmr #<face>")
        face = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(face, Face), f"{tokens[0]} is not a face")
        ring = eu.kfmr(self.kernel, face)
        self._out(f"KFMR: killed face #{face.oid}, its loop #{ring.oid} became "
                  f"a ring of the adjacent face (-1F +1R, genus+1)")

    def _micro_mfkr(self, tokens: List[str]) -> None:
        _expect(len(tokens) == 1, "usage: micro mfkr #<loop>")
        loop = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(loop, Loop), f"{tokens[0]} is not a loop")
        face = eu.mfkr(self.kernel, loop)
        self._out(f"MFKR: promoted ring loop #{loop.oid} into face #{face.oid} "
                  f"(+1F -1R, genus-1)")

    # ------------------------------------------------------------------ #
    # 4.2  Macro modeling commands
    # ------------------------------------------------------------------ #
    def do_create(self, arg: str) -> None:
        "create box|sphere|cylinder|nurbs|plane <params> [as #id|@name]  - build a primitive solid"
        tokens = shlex.split(arg)
        _expect(bool(tokens),
                "usage: create box|sphere|cylinder|nurbs|plane ... [as #id|@name]")
        kind = tokens[0].lower()
        rest, as_spec = self._split_as(tokens[1:])
        oid = self._explicit_id(as_spec)

        if kind == "box":
            _expect(len(rest) == 3, "usage: create box <L> <W> <H> [as #id|@name]")
            l, w, h = (float(t) for t in rest)
            solid = macro.create_box(self.kernel, l, w, h, solid_oid=oid)
            detail = f"box {l}x{w}x{h}"
        elif kind == "sphere":
            _expect(len(rest) in (1, 3),
                    "usage: create sphere <radius> [slices stacks] [as #id|@name]")
            radius = float(rest[0])
            slices = int(rest[1]) if len(rest) == 3 else 16
            stacks = int(rest[2]) if len(rest) == 3 else 8
            solid = macro.create_sphere(self.kernel, radius, slices, stacks,
                                        solid_oid=oid)
            detail = f"sphere r={radius} ({slices}x{stacks})"
        elif kind == "cylinder":
            _expect(len(rest) in (2, 3),
                    "usage: create cylinder <radius> <height> [slices] [as #id|@name]")
            radius, height = float(rest[0]), float(rest[1])
            slices = int(rest[2]) if len(rest) == 3 else 16
            solid = macro.create_cylinder(self.kernel, radius, height, slices,
                                          solid_oid=oid)
            detail = f"cylinder r={radius} h={height} ({slices})"
        elif kind == "nurbs":
            _expect(len(rest) == 2,
                    "usage: create nurbs <size> <height> [as #id|@name]")
            size, height = float(rest[0]), float(rest[1])
            solid = macro.create_nurbs_dome(self.kernel, size, height, solid_oid=oid)
            detail = f"nurbs dome size={size} h={height}"
        elif kind == "plane":
            _expect(len(rest) == 2,
                    "usage: create plane <width> <height> [as #id|@name]")
            width, height = float(rest[0]), float(rest[1])
            solid = macro.create_plane(self.kernel, width, height, solid_oid=oid)
            detail = f"plane {width}x{height}"
        else:
            raise CliError(f"unknown primitive '{kind}' "
                           "(supported: box, sphere, cylinder, nurbs, plane)")

        self._bind_alias(as_spec, solid)
        self._out(view.format_entity_created("Solid", solid.oid, detail))

    def do_extrude(self, arg: str) -> None:
        "extrude #<face> <dx> <dy> <dz>  - sweep a planar face along a vector into a prism"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 4, "usage: extrude #<face> <dx> <dy> <dz>")
        face = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(face, Face), f"#{tokens[0]} is not a face")
        dx, dy, dz = (float(t) for t in tokens[1:])
        macro.extrude(self.kernel, face, Point3D(dx, dy, dz))
        self._out(f"extruded face #{face.oid} by ({dx}, {dy}, {dz})")

    def do_revolve(self, arg: str) -> None:
        "revolve #<face> <x|y|z> <angle> [segs]  - faceted rotational sweep"
        tokens = shlex.split(arg)
        _expect(len(tokens) in (3, 4), "usage: revolve #<face> <x|y|z> <angle> [segments]")
        face = self.kernel.get(self._resolve_id(tokens[0]))
        _expect(isinstance(face, Face), f"#{tokens[0]} is not a face")
        axis = tokens[1].lower()
        angle = float(tokens[2])
        segments = int(tokens[3]) if len(tokens) == 4 else 4
        macro.revolve(self.kernel, face, axis, angle, segments)
        self._out(f"revolved face #{face.oid} about {axis} by {angle} deg")

    def do_trim(self, arg: str) -> None:
        "trim curve|surface|#<solid> ...  - split a curve, crop a NURBS face, or half-space cut a solid"
        tokens = shlex.split(arg)
        _expect(len(tokens) >= 1, "usage: trim curve|surface|<solid> ...")
        kind = tokens[0].lower()

        if kind == "curve":
            _expect(len(tokens) == 4 and tokens[2].lower() == "at",
                    "usage: trim curve #<edge> at <u>")
            edge = self.kernel.get(self._resolve_id(tokens[1]))
            _expect(isinstance(edge, Edge), f"#{tokens[1]} is not an edge")
            u = float(tokens[3])
            v, new_edge = macro.trim_curve(self.kernel, edge, u)
            self._out(f"trimmed edge #{edge.oid} at u={u}: new vertex #{v.oid}, "
                      f"new edge #{new_edge.oid}")

        elif kind == "surface":
            _expect(
                len(tokens) >= 3,
                "usage: trim surface #<face> keep <u0> <u1> <v0> <v1>  "
                "(or legacy: trim surface #<face> by #<loop>)",
            )
            face = self.kernel.get(self._resolve_id(tokens[1]))
            _expect(isinstance(face, Face), f"#{tokens[1]} is not a face")
            sub = tokens[2].lower()
            if sub == "keep":
                _expect(len(tokens) == 7,
                        "usage: trim surface #<face> keep <u0> <u1> <v0> <v1>")
                u0, u1, v0, v1 = (float(t) for t in tokens[3:7])
                macro.trim_surface_region(self.kernel, face, u0, u1, v0, v1)
                self._out(f"trimmed surface of face #{face.oid} to "
                          f"u=[{u0},{u1}] v=[{v0},{v1}]")
            elif sub == "by":
                _expect(len(tokens) == 4,
                        "usage: trim surface #<face> by #<loop>")
                loop_id = self._resolve_id(tokens[3])
                macro.trim_surface(self.kernel, face, loop_id)
                self._out(f"trim boundary #{loop_id} tagged on face #{face.oid}")
            else:
                _expect(False,
                        "usage: trim surface #<face> keep <u0> <u1> <v0> <v1>  "
                        "(or: by #<loop>)")

        elif len(tokens) >= 3 and tokens[1].lower() == "by" \
                and tokens[2].lower() == "surface":
            # trim #<solid> by surface #<face> [keep above|below]
            _expect(
                len(tokens) == 4
                or (len(tokens) == 6 and tokens[4].lower() == "keep"),
                "usage: trim #<solid> by surface #<face> [keep above|below]",
            )
            entity = self.kernel.get(self._resolve_id(tokens[0]))
            solid = self.kernel.solid_of(entity)
            _expect(solid is not None, f"#{entity.oid} is not part of any solid")
            cutter_face = self.kernel.get(self._resolve_id(tokens[3]))
            _expect(isinstance(cutter_face, Face),
                    f"cutter #{tokens[3]} is not a face")
            keep_below = False
            if len(tokens) == 6:
                side = tokens[5].lower()
                _expect(side in ("above", "below", "+", "-"),
                        "keep side must be 'above' or 'below'")
                keep_below = side in ("below", "-")
            result = macro.trim_solid_by_surface(
                self.kernel, solid, cutter_face, keep_below=keep_below)
            self._out(
                f"trim solid #{solid.oid} by NURBS surface of face "
                f"#{cutter_face.oid} keep={'below' if keep_below else 'above'} "
                f"(above = the cutter-normal side):\n"
                f"  keep={result.n_keep}  cut={result.n_cut}  discard={result.n_discard}"
            )

        else:
            # trim #<solid> by plane <nx> <ny> <nz> <d> [keep above|below]
            _expect(
                (len(tokens) == 7
                 or (len(tokens) == 9 and tokens[7].lower() == "keep"))
                and tokens[1].lower() == "by"
                and tokens[2].lower() == "plane",
                "usage: trim #<solid> by plane <nx> <ny> <nz> <d> [keep above|below]"
                "  |  trim #<solid> by surface #<face> [keep above|below]",
            )
            entity = self.kernel.get(self._resolve_id(tokens[0]))
            solid = self.kernel.solid_of(entity)
            _expect(solid is not None,
                    f"#{entity.oid} is not part of any solid")
            nx, ny, nz, d = (float(t) for t in tokens[3:7])
            keep_below = False
            if len(tokens) == 9:
                side = tokens[8].lower()
                _expect(side in ("above", "below", "+", "-"),
                        "keep side must be 'above' or 'below'")
                keep_below = side in ("below", "-")
            result = macro.trim_solid_by_plane(
                self.kernel, solid, nx, ny, nz, d, keep_below=keep_below)
            self._out(
                f"trim solid #{solid.oid} by plane ({nx},{ny},{nz})·P={d} "
                f"keep={'below' if keep_below else 'above'}:\n"
                f"  keep={result.n_keep}  cut={result.n_cut}  discard={result.n_discard}"
            )

    def do_extend(self, arg: str) -> None:
        "extend #<edge|face> to plane <nx> <ny> <nz> <d> | to #<face> [along <dx> <dy> <dz>]"
        tokens = shlex.split(arg)
        _expect(
            len(tokens) >= 3 and tokens[1].lower() == "to",
            "usage: extend #<edge|face> to plane <nx> <ny> <nz> <d>  |  "
            "extend #<edge|face> to #<face> [along <dx> <dy> <dz>]",
        )
        source = self.kernel.get(self._resolve_id(tokens[0]))
        rest = tokens[2:]

        # Optional trailing 'along <dx> <dy> <dz>' (sweep direction for faces).
        along: Optional[Point3D] = None
        lowered = [t.lower() for t in rest]
        if "along" in lowered:
            idx = lowered.index("along")
            dirn = rest[idx + 1:idx + 4]
            _expect(len(dirn) == 3, "usage: ... along <dx> <dy> <dz>")
            along = Point3D(float(dirn[0]), float(dirn[1]), float(dirn[2]))
            rest = rest[:idx]

        # Resolve the target into ('plane', TrimPlane) or ('surface', NURBSSurface).
        if rest and rest[0].lower() == "plane":
            _expect(len(rest) == 5, "usage: ... to plane <nx> <ny> <nz> <d>")
            nx, ny, nz, d = (float(t) for t in rest[1:5])
            target = ("plane", TrimPlane(Point3D(nx, ny, nz), d))
            target_desc = f"plane ({nx},{ny},{nz})·P={d}"
        else:
            _expect(len(rest) == 1, "usage: ... to #<face>  (target must be a face)")
            tface = self.kernel.get(self._resolve_id(rest[0]))
            _expect(isinstance(tface, Face), f"target #{rest[0]} is not a face")
            surf = getattr(tface, "surface", None)
            if isinstance(surf, NURBSSurface):
                target = ("surface", surf)
                target_desc = f"NURBS face #{tface.oid}"
            else:
                target = ("plane", macro.face_plane(tface))
                target_desc = f"plane of face #{tface.oid}"

        if isinstance(source, Edge):
            v, new_edge = macro.extend_curve(self.kernel, source, target)
            self._out(
                f"extended edge #{source.oid} to {target_desc}: "
                f"new vertex #{v.oid} at {v.point}, new edge #{new_edge.oid}"
            )
        elif isinstance(source, Face):
            macro.extend_face(self.kernel, source, target, direction=along)
            solid = self.kernel.solid_of(source)
            self._out(
                f"extended face #{source.oid} to {target_desc} "
                f"(swept new cap on the target): "
                f"V={solid.num_vertices} E={solid.num_edges} F={solid.num_faces}"
            )
        else:
            _expect(False, f"#{tokens[0]} is not an edge or a face")

    def do_intersect(self, arg: str) -> None:
        "intersect #<faceA> #<faceB> [samples <n>] [as @name]  - NURBS/NURBS intersection curve as a wire"
        tokens = shlex.split(arg)
        tokens, as_spec = self._split_as(tokens)
        samples = 32
        lowered = [t.lower() for t in tokens]
        if "samples" in lowered:
            idx = lowered.index("samples")
            _expect(idx + 1 < len(tokens), "usage: ... samples <n>")
            samples = int(tokens[idx + 1])
            tokens = tokens[:idx] + tokens[idx + 2:]
        _expect(len(tokens) == 2,
                "usage: intersect #<faceA> #<faceB> [samples <n>] [as @name]")
        face_a = self.kernel.get(self._resolve_id(tokens[0]))
        face_b = self.kernel.get(self._resolve_id(tokens[1]))
        _expect(isinstance(face_a, Face) and isinstance(face_b, Face),
                "intersect needs two faces (each carrying a NURBS surface)")
        wire, closed, n = macro.intersect_surfaces(
            self.kernel, face_a, face_b, samples=samples)
        self._bind_alias(as_spec, wire)
        self._out(
            f"+ created intersection wire Solid #{wire.oid}: "
            f"{'closed loop' if closed else 'open curve'} of {n} points\n"
            f"  each vertex stores (u,v) on BOTH surfaces "
            f"(disp math #<vertex>); branch data on faces "
            f"#{face_a.oid}/#{face_b.oid} (disp math)"
        )

    def do_blend(self, arg: str) -> None:
        "blend #<faceA> #<faceB> width <w> [samples <n>] [as @name]  - G2 blend patch across the intersection"
        tokens = shlex.split(arg)
        tokens, as_spec = self._split_as(tokens)
        lowered = [t.lower() for t in tokens]
        _expect("width" in lowered,
                "usage: blend #<faceA> #<faceB> width <w> [samples <n>] [as @name]")
        wi = lowered.index("width")
        _expect(wi + 1 < len(tokens), "usage: ... width <w>")
        width = float(tokens[wi + 1])
        rest = tokens[:wi] + tokens[wi + 2:]
        samples = 9
        lowered = [t.lower() for t in rest]
        if "samples" in lowered:
            si = lowered.index("samples")
            _expect(si + 1 < len(rest), "usage: ... samples <n>")
            samples = int(rest[si + 1])
            rest = rest[:si] + rest[si + 2:]
        _expect(len(rest) == 2,
                "usage: blend #<faceA> #<faceB> width <w> [samples <n>] [as @name]")
        face_a = self.kernel.get(self._resolve_id(rest[0]))
        face_b = self.kernel.get(self._resolve_id(rest[1]))
        _expect(isinstance(face_a, Face) and isinstance(face_b, Face),
                "blend needs two faces (each carrying a NURBS surface)")
        solid = macro.blend_surfaces(self.kernel, face_a, face_b,
                                     width=width, samples=samples)
        self._bind_alias(as_spec, solid)
        patch = next(f.surface for f in solid.faces if f.surface)
        self._out(
            f"+ created blend Solid #{solid.oid}: quintic Hermite patch "
            f"(net {patch.n_u}x{patch.n_v}, degree {patch.degree_u}x"
            f"{patch.degree_v}) matching position + 1st + 2nd derivative "
            f"on both rails (curvature-continuous join)"
        )

    def do_delete(self, arg: str) -> None:
        "delete #<solid>  - remove a solid and everything it owns"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 1, "usage: delete #<solid>")
        entity = self.kernel.get(self._resolve_id(tokens[0]))
        solid = self.kernel.solid_of(entity)
        _expect(solid is not None, f"#{tokens[0]} is not part of any solid")
        oid, name = solid.oid, solid.name
        self.kernel.delete_solid(solid)
        # Drop aliases that pointed into the deleted solid.
        dead = [a for a, target in self.aliases.items()
                if self.kernel.find(target) is None]
        for a in dead:
            del self.aliases[a]
        self._out(f"- deleted Solid #{oid} '{name}'"
                  + (f" (unbound @{', @'.join(dead)})" if dead else ""))

    def do_view(self, arg: str) -> None:
        "view [solid|wire|points] [#id|@alias|$solid]  - open 3-D interactive viewer"
        tokens = shlex.split(arg)

        MODE_ALIASES = {"wireframe": "wire", "shaded": "solid", "mesh": "solid",
                        "surf": "solid", "solid": "solid", "wire": "wire",
                        "points": "points", "pt": "points"}
        mode = "solid"
        solid_token: Optional[str] = None

        if tokens and tokens[0].lower() in MODE_ALIASES:
            mode = MODE_ALIASES[tokens[0].lower()]
            tokens = tokens[1:]
        if tokens:
            solid_token = tokens[0]

        solid = self._resolve_solid([solid_token] if solid_token else [])
        if solid is None:
            _expect(bool(self.kernel.solids), "no solids to view")
            solid = self.kernel.solids[-1]

        try:
            from .viewer import show_solid
            self._out(f"opening viewer for solid #{solid.oid} '{solid.name}' "
                      f"(mode={mode}) ...")
            show_solid(solid, mode=mode)
        except (ImportError, RuntimeError) as exc:
            raise CliError(str(exc))

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def do_setpoint(self, arg: str) -> None:
        "setpoint #<vertex> as (<x>, <y>, <z>)  - update coordinates"
        head, coords = self._split_on_keyword(arg, "as")
        v = self.kernel.get(self._resolve_id(head[0]))
        _expect(isinstance(v, Vertex), "setpoint target must be a vertex")
        nums = self._parse_coords(coords)
        _expect(len(nums) == 3, "expected 3 coordinates")
        v.point = Point3D(*nums)
        self._out(f"vertex #{v.oid} -> {v.point}")

    def do_setcurve(self, arg: str) -> None:
        "setcurve #<edge> as bezier #<cp1> ... #<cpN>  - attach a Bezier"
        head, tail = self._split_on_keyword(arg, "as")
        edge = self.kernel.get(self._resolve_id(head[0]))
        _expect(isinstance(edge, Edge), "setcurve target must be an edge")
        _expect(bool(tail) and tail[0].lower() == "bezier",
                "usage: setcurve #<edge> as bezier #<cp1> ... #<cpN>")
        cp_ids = tail[1:]
        _expect(len(cp_ids) >= 2, "a Bezier needs at least 2 control points")
        points = []
        for cid in cp_ids:
            vtx = self.kernel.get(self._resolve_id(cid))
            _expect(isinstance(vtx, Vertex), f"#{cid} is not a vertex")
            points.append(vtx.point)
        edge.curve = Bezier(points)
        self._out(f"edge #{edge.oid} <- Bezier degree {edge.curve.degree}")

    def do_setsurface(self, arg: str) -> None:
        "setsurface #<face> as nurbs <deg_u> <deg_v>  - attach a NURBS patch"
        head, tail = self._split_on_keyword(arg, "as")
        face = self.kernel.get(self._resolve_id(head[0]))
        _expect(isinstance(face, Face), "setsurface target must be a face")
        _expect(len(tail) == 3 and tail[0].lower() == "nurbs",
                "usage: setsurface #<face> as nurbs <deg_u> <deg_v>")
        deg_u, deg_v = int(tail[1]), int(tail[2])
        face.surface = self._build_surface(face, deg_u, deg_v)
        self._out(f"face #{face.oid} <- NURBS degree ({deg_u}, {deg_v})")

    def _build_surface(self, face: Face, deg_u: int, deg_v: int) -> NURBSSurface:
        pts = [he.vertex.point for he in face.outer.halfedges()]
        n_u, n_v = deg_u + 1, deg_v + 1
        if len(pts) >= n_u * n_v:
            net = [pts[i * n_v:(i + 1) * n_v] for i in range(n_u)]
            return NURBSSurface(net, deg_u, deg_v)
        # Fall back to a bilinear patch from the first four corners.
        _expect(len(pts) >= 4, "face needs at least 4 vertices for a NURBS patch")
        net = [[pts[0], pts[1]], [pts[3], pts[2]]]
        return NURBSSurface(net, degree_u=1, degree_v=1)

    # ------------------------------------------------------------------ #
    # 4.4  Spatial transformations
    # ------------------------------------------------------------------ #
    def do_move(self, arg: str) -> None:
        "move #<entity> <dx> <dy> <dz>  - translate"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 4, "usage: move #<entity> <dx> <dy> <dz>")
        entity = self.kernel.get(self._resolve_id(tokens[0]))
        dx, dy, dz = (float(t) for t in tokens[1:])
        self._apply_transform(entity, translation_matrix(dx, dy, dz))
        self._out(f"moved #{entity.oid} by ({dx}, {dy}, {dz})")

    def do_rotate(self, arg: str) -> None:
        "rotate #<entity> <x|y|z> <angle>  - rotate about a principal axis"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 3, "usage: rotate #<entity> <x|y|z> <angle>")
        entity = self.kernel.get(self._resolve_id(tokens[0]))
        axis, angle = tokens[1].lower(), float(tokens[2])
        self._apply_transform(entity, rotation_matrix(axis, angle))
        self._out(f"rotated #{entity.oid} about {axis} by {angle} deg")

    def do_scale(self, arg: str) -> None:
        "scale #<entity> <factor>  - uniform scale about the centroid"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 2, "usage: scale #<entity> <factor>")
        entity = self.kernel.get(self._resolve_id(tokens[0]))
        factor = float(tokens[1])
        verts = self._vertices_of(entity)
        center = self._centroid(verts)
        self._apply_transform(entity, scaling_matrix(factor, center))
        self._out(f"scaled #{entity.oid} by {factor}")

    def _vertices_of(self, entity) -> List[Vertex]:
        if isinstance(entity, Vertex):
            return [entity]
        if isinstance(entity, Edge):
            return [entity.he1.vertex, entity.he1.end_vertex]
        if isinstance(entity, Face):
            seen, out = set(), []
            for loop in entity.loops:
                for v in loop.vertices():
                    if v.oid not in seen:
                        seen.add(v.oid)
                        out.append(v)
            return out
        if isinstance(entity, Solid):
            return list(entity.vertices)
        raise CliError(f"#{entity.oid} cannot be transformed")

    def _apply_transform(self, entity, matrix) -> None:
        # Topology side: move every owned vertex.
        for v in self._vertices_of(entity):
            if v.point is not None:
                v.point = apply_matrix(matrix, v.point)
        # Geometry side: the attached NURBS control nets and Bezier control
        # points must follow, or the surface/curve would detach from its
        # boundary (topology and geometry always update together).
        faces: List[Face] = []
        edges: List[Edge] = []
        if isinstance(entity, Solid):
            faces = list(entity.faces)
            edges = list(entity.edges)
        elif isinstance(entity, Face):
            faces = [entity]
            edges = [he.edge for lp in entity.loops
                     for he in lp.halfedges() if he.edge]
        elif isinstance(entity, Edge):
            edges = [entity]
        for f in faces:
            surf = getattr(f, "surface", None)
            if isinstance(surf, NURBSSurface):
                surf.control_net = [[apply_matrix(matrix, p) for p in row]
                                    for row in surf.control_net]
        seen = set()
        for e in edges:
            if e is None or e.oid in seen:
                continue
            seen.add(e.oid)
            curve = getattr(e, "curve", None)
            if isinstance(curve, Bezier):
                curve.control_points = [apply_matrix(matrix, p)
                                        for p in curve.control_points]

    @staticmethod
    def _centroid(verts: List[Vertex]) -> Point3D:
        pts = [v.point for v in verts if v.point is not None]
        if not pts:
            return Point3D(0, 0, 0)
        sx = sum(p.x for p in pts) / len(pts)
        sy = sum(p.y for p in pts) / len(pts)
        sz = sum(p.z for p in pts) / len(pts)
        return Point3D(sx, sy, sz)

    # ------------------------------------------------------------------ #
    # 4.5  Information & debugging
    # ------------------------------------------------------------------ #
    def do_disp(self, arg: str) -> None:
        "disp topology|math|vertices [#id]  - inspect entities"
        tokens = shlex.split(arg)
        _expect(bool(tokens), "usage: disp topology|math|vertices [#id]")
        what = tokens[0].lower()
        rest = tokens[1:]
        if what == "topology":
            solids = [self._resolve_solid(rest)] if rest else self.kernel.solids
            _expect(bool(solids) and solids[0] is not None, "no solids to display")
            for s in solids:
                self._out(view.format_topology(s))
        elif what == "vertices":
            solids = [self._resolve_solid(rest)] if rest else self.kernel.solids
            for s in solids:
                self._out(view.format_vertices(s))
        elif what == "math":
            _expect(len(rest) == 1, "usage: disp math #<id>")
            entity = self.kernel.get(self._resolve_id(rest[0]))
            self._out(view.format_math(entity))
        else:
            raise CliError("usage: disp topology|math|vertices [#id]")

    def do_check(self, arg: str) -> None:
        "check validity [#id]  - run the Euler + pointer integrity checks"
        tokens = shlex.split(arg)
        _expect(bool(tokens) and tokens[0].lower() == "validity",
                "usage: check validity [#id]")
        rest = tokens[1:]
        solids = [self._resolve_solid(rest)] if rest else self.kernel.solids
        _expect(bool(solids) and solids[0] is not None, "no solids to check")
        for s in solids:
            self._out(view.format_validation(check_solid(s)))

    def do_set(self, arg: str) -> None:
        "set @<name> <#id|$var|@alias>  - bind a symbolic name to an entity"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 2, "usage: set @<name> <#id|$var|@alias>")
        name = tokens[0]
        _expect(name.startswith("@"), "alias name must start with '@'")
        oid = self._resolve_id(tokens[1])
        self.kernel.get(oid)  # ensure the target exists
        self.aliases[name[1:]] = oid
        self._out(f"@{name[1:]} -> #{oid}")

    def do_unset(self, arg: str) -> None:
        "unset @<name>  - remove a symbolic name"
        tokens = shlex.split(arg)
        _expect(len(tokens) == 1 and tokens[0].startswith("@"),
                "usage: unset @<name>")
        self.aliases.pop(tokens[0][1:], None)
        self._out(f"removed @{tokens[0][1:]}")

    def do_vars(self, arg: str) -> None:
        "vars  - list @aliases and the current $last-created entities"
        if self.aliases:
            self._out("aliases:")
            for n, o in sorted(self.aliases.items()):
                self._out(f"  @{n} -> #{o}")
        else:
            self._out("aliases: (none)")
        self._out("$last entities:")
        any_last = False
        for kind in ("solid", "vertex", "edge", "face", "loop"):
            e = self.kernel.registry.last(kind)
            if e is not None:
                any_last = True
                self._out(f"  ${kind} -> #{e.oid}")
        if not any_last:
            self._out("  (none yet)")

    def do_list(self, arg: str) -> None:
        "list  - summarize every solid in the session"
        if not self.kernel.solids:
            self._out("(no solids)")
            return
        for s in self.kernel.solids:
            self._out(f"Solid #{s.oid} '{s.name}'  "
                      f"V={s.num_vertices} E={s.num_edges} F={s.num_faces}")

    # ------------------------------------------------------------------ #
    # 4.6  Testing & I/O
    # ------------------------------------------------------------------ #
    def do_run(self, arg: str) -> None:
        'run "<script.txt>"  - execute commands line-by-line from a file'
        tokens = shlex.split(arg)
        _expect(len(tokens) == 1, 'usage: run "<script_path>"')
        path = tokens[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:
            raise CliError(f"cannot open script: {exc}")
        self._out(f"--- running {path} ---")
        for n, raw in enumerate(lines, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith(("#", "//")):
                continue
            self._out(f"[{n}] {line}")
            self.onecmd(line)
        self._out(f"--- finished {path} ---")

    def do_save(self, arg: str) -> None:
        'save "<file.step>" [#solid] [faceted]  - export a solid to STEP'
        tokens = shlex.split(arg)
        faceted = False
        if tokens and tokens[-1].lower() == "faceted":
            faceted = True
            tokens = tokens[:-1]
        _expect(len(tokens) in (1, 2),
                'usage: save "<file.step>" [#solid] [faceted]')
        path = tokens[0]
        solid = self._resolve_solid(tokens[1:]) if len(tokens) == 2 else None
        if solid is None:
            _expect(bool(self.kernel.solids), "no solids to save")
            solid = self.kernel.solids[0]
        stepio.save(solid, path, faceted=faceted)
        mode = " (faceted trim shells)" if faceted else ""
        self._out(f"saved solid #{solid.oid} -> {path}{mode}")

    def do_load(self, arg: str) -> None:
        'load "<file.step>"  - import points from a STEP file'
        tokens = shlex.split(arg)
        _expect(len(tokens) == 1, 'usage: load "<file.step>"')
        solid = stepio.load(self.kernel, tokens[0])
        self._out(f"loaded -> {solid.name} (solid #{solid.oid})")

    # ------------------------------------------------------------------ #
    # Exit
    # ------------------------------------------------------------------ #
    def do_exit(self, arg: str) -> bool:
        "exit  - leave the REPL"
        self._out("bye.")
        return True

    do_quit = do_exit
    do_EOF = do_exit

    # ------------------------------------------------------------------ #
    # 4.7  Help topics  (called by 'help <topic>')
    # ------------------------------------------------------------------ #

    _HR = "-" * 56

    def _h(self, text: str) -> None:
        """Print help text, stripping one level of leading indentation."""
        import textwrap
        self._out(textwrap.dedent(text).strip())

    # ── create & each primitive ──────────────────────────────────────── #

    def help_create(self) -> None:
        self._h(f"""
        create -- build a solid primitive
        {self._HR}
        Syntax:  create <kind> <params> [as #id|@name]

          create box     <L> <W> <H>
          create sphere  <radius> [<slices> <stacks>]
          create cylinder <radius> <height> [<slices>]
          create nurbs   <size> <height>
          create plane   <width> <height>

        Optional suffix:
          as @name   alias for later use   (e.g.  as @box1)
          as #id     request a specific entity id

        Type  'help box', 'help sphere', 'help cylinder',
              'help nurbs', or 'help plane'  for parameter details.

        Quick examples:
          create box 10 20 30 as @b
          create sphere 5 as @s
          create sphere 5 16 8
          create cylinder 4 12 24 as @c
          create nurbs 20 8 as @n
          create plane 30 15 as @p
        """)

    def help_box(self) -> None:
        self._h(f"""
        create box <L> <W> <H> [as #id|@name]
        {self._HR}
        Build an axis-aligned rectangular box of dimensions:
          <L>  length along X  (required)
          <W>  width  along Y  (required)
          <H>  height along Z  (required)

        Origin of the box is at (0, 0, 0).
        Result: V=8, E=12, F=6  (closed solid)

        Examples:
          create box 10 20 30             // 10×20×30 box
          create box 5 5 5 as @cube       // unit-ish cube with alias
          create box 100 50 25 as @slab
        """)

    def help_sphere(self) -> None:
        self._h(f"""
        create sphere <radius> [<slices> <stacks>] [as #id|@name]
        {self._HR}
        Build a faceted UV sphere centred at the origin.
          <radius>  sphere radius            (required)
          <slices>  longitude divisions      (optional, default 16, min 3)
          <stacks>  latitude divisions       (optional, default 8,  min 2)

        slices and stacks must be given together or omitted together.
        Result: closed manifold solid; poles are triangle fans,
                mid-bands are quads.

        Examples:
          create sphere 10                 // r=10, 16 slices × 8 stacks
          create sphere 5 as @s            // r=5 with alias
          create sphere 5 24 12 as @hires  // high-resolution sphere
          create sphere 1 8 4              // coarse sphere for testing
        """)

    def help_cylinder(self) -> None:
        self._h(f"""
        create cylinder <radius> <height> [<slices>] [as #id|@name]
        {self._HR}
        Build a capped cylinder of given radius and height along +Z.
          <radius>  base/top circle radius   (required)
          <height>  cylinder height          (required)
          <slices>  number of side faces     (optional, default 16, min 3)

        Result: closed solid with flat caps; bottom at z=0, top at z=height.

        Examples:
          create cylinder 5 20             // r=5, h=20, 16 sides
          create cylinder 5 20 as @cyl     // with alias
          create cylinder 4 12 24 as @c    // 24-sided cylinder
          create cylinder 3 8 6            // hexagonal prism (6 sides)
        """)

    def help_nurbs(self) -> None:
        self._h(f"""
        create nurbs <size> <height> [as #id|@name]
        {self._HR}
        Build a square degree-2 NURBS dome lamina.
          <size>    side length of the square footprint  (required)
                    The lamina spans from -size/2 to +size/2 in both X and Y.
          <height>  apex height of the dome above the XY-plane  (required)

        The result is a 2-face lamina (front face + back face sharing 4 edges).
        The front face carries a 3×3 B-spline (NURBS) control net:
          corners   → z = 0  (on the base plane)
          mid-edges → z = height/2
          centre    → z = height

        Result: V=4, E=4, F=2
        Exports as B_SPLINE_SURFACE_WITH_KNOTS in STEP.

        Examples:
          create nurbs 20 8              // 20×20 dome, height=8
          create nurbs 20 8 as @dome
          create nurbs 10 3 as @small

        Trim workflow:
          create nurbs 20 8 as @n
          trim @n by plane 1 0 0 0       // keep x>0 half
          save "half_dome.step" @n
        """)

    def help_plane(self) -> None:
        self._h(f"""
        create plane <width> <height> [as #id|@name]
        {self._HR}
        Build a flat rectangular lamina centred at the origin in the XY-plane.
          <width>   extent along X  (required)
          <height>  extent along Y  (required)

        Vertices are placed at (±width/2, ±height/2, 0).
        The lamina has 2 faces sharing 4 boundary edges.
        Result: V=4, E=4, F=2

        Examples:
          create plane 20 20              // 20×20 square lamina
          create plane 20 20 as @pl       // with alias
          create plane 30 15 as @rect     // 30×15 rectangle

        Trim workflow:
          create plane 20 20 as @p
          trim @p by plane 0 1 0 3        // cut at y=3; keep y>3 half
          check validity @p               // V=6 E=8 F=4  Euler OK
          save "trimmed.step" @p          // OPEN_SHELL with 2 keep faces
        """)

    # ── modeling commands ────────────────────────────────────────────── #

    def help_extrude(self) -> None:
        self._h(f"""
        extrude #<face> <dx> <dy> <dz>
        {self._HR}
        Sweep a planar face along the vector (dx, dy, dz) to build a prism.
          #<face>      the face to sweep (use $face, @alias, or #id)
          <dx dy dz>   extrusion direction and length

        The passed face becomes the top cap; a new bottom cap is also created.
        One side face is generated per boundary edge of the profile.

        Examples:
          create box 10 10 0.01 as @base    // make a thin base
          extrude $face 0 0 10              // extrude last face 10 units up

          micro mvfs 0 0 0
          micro mev $vertex 5 0 0
          micro mev $vertex 5 5 0
          micro mev $vertex 0 5 0
          micro mef $vertex #<first_v>
          extrude $face 0 0 8              // extrude the quad 8 units
        """)

    def help_revolve(self) -> None:
        self._h(f"""
        revolve #<face> <x|y|z> <angle_deg> [<segments>]
        {self._HR}
        Sweep a planar profile around a principal axis to produce a
        faceted solid of revolution.
          #<face>       the profile face to revolve
          <x|y|z>       axis of revolution
          <angle_deg>   sweep angle in degrees (e.g. 360 for full revolution)
          <segments>    number of angular steps (optional, default 4)

        Each side face is tagged with a bilinear NURBS patch (cylindrical approx).
        This is a teaching-grade revolve: topology is correct but the surface
        math is faceted, not an exact analytic surface.

        Examples:
          create nurbs 10 0.1 as @profile
          revolve $face y 360 16          // full revolution, 16 segments

          revolve $face z 180 8           // 180° half-revolution, 8 segments
        """)

    def help_trim(self) -> None:
        self._h(f"""
        trim -- parametric/topological cut operations
        {self._HR}
        The part that SURVIVES is always named explicitly by a 'keep' clause
        (except the curve form, which is a pure split and keeps both halves).

        Forms:

        1)  trim curve #<edge> at <u>
            SPLIT an edge at parameter u in (0,1) -- inserts a vertex M so the edge
            becomes two connected segments A-M and M-B. This keeps BOTH halves
            (it does not discard anything). A Bezier curve, if attached, is
            geometrically subdivided at u.
              <u>  split parameter strictly between 0 and 1

        2a) trim surface #<face> keep <u0> <u1> <v0> <v1>
            Crop a NURBS face to the parametric window you KEEP: [u0,u1] x [v0,v1].
            The surface is actually resized (view / save show only that window).
            Requires 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1.

        2b) trim surface #<face> by #<loop_id>
            LEGACY, metadata only: tags the face with a loop id for 'disp math'.
            It does NOT change geometry. For a real surface trim use form 2a.

        3)  trim #<solid> by plane <nx> <ny> <nz> <d> [keep above|below]
            Half-space cut by the plane  nx*x + ny*y + nz*z = d.
            The 'keep' clause chooses which half SURVIVES (default 'above'):
              keep above   retain nx*x+ny*y+nz*z > d  (the plane-normal side)
              keep below   retain nx*x+ny*y+nz*z < d  (the opposite half)
            'below' only flips which side survives -- type the reference plane and
            its normal exactly as-is (no need to negate nx..d yourself).

            TOPOLOGICAL trim for ALL solids (lamina, box, sphere, cylinder):
              * split_edge inserts exact intersection vertices on crossing edges
                (an edge carrying a Bezier is cut ON its curve, not the chord)
              * MEF splits each straddling face into keep / discard halves
              * an interior CAP cut (the surface crosses, the flat boundary does
                not) extracts the section curve in the surface's (u,v) space and
                lifts it into the topology: a cap face + inner RING via
                MEV-bridge / MEV-chain / MEF / KEMR; every section vertex lies
                on the true surface and stores its (u,v) (see disp math)
              * discard-side faces are flagged; NURBS is cropped to the keep half
              * topology stays a valid manifold (Euler V-E+F-R invariant holds)
              * 'save' writes an OPEN_SHELL of the surviving faces
            A cut through existing vertices (e.g. sphere at its equator) needs no
            edge split; those vertices form the section boundary directly.
            Only a plane that misses the solid falls back to parametric metadata.

        4)  trim #<solid> by surface #<face> [keep above|below]
            Half-space cut by a CURVED NURBS surface (NURBS x NURBS trim).
            The cutter face's surface becomes a signed-distance field
            (closest-point projection + normal sign), and the same topological
            pipeline runs against it: crossing edges are bisected onto the
            curved cutter, straddling faces MEF-cut, interior sections become
            rings. 'keep above' = the side the cutter's NORMAL points to
            (note: 'create nurbs' domes have a -z normal), 'keep below' the
            other side.

              // box carved by a dome surface (keep the part above the dome)
              create nurbs 20 8 as @dome
              set @cutter $face
              create box 8 8 8 as @bx
              move @bx -4 -4 0
              trim @bx by surface @cutter keep below
              check validity @bx

        Arrange the plane so it slices THROUGH the solid (perpendicular or
        oblique), not parallel to a face or merely grazing it.

        Examples (each block is self-contained -- copy/paste and run as-is):

          // 1) Plane lamina cut by the perpendicular plane y=3, keep y>3
          create plane 20 20 as @p
          trim @p by plane 0 1 0 3
          check validity @p
          save "trim_plane.step" @p

          // 2) Box: OBLIQUE cut x+y+z=15 lops off the far corner at an angle
          create box 10 10 10 as @b
          trim @b by plane 1 1 1 15

          // 3) Cylinder: SLANTED cut z=0.6x+6 gives an elliptical top
          create cylinder 6 12 24 as @cy
          trim @cy by plane -0.6 0 1 6

          // 4) Sphere: tilted cap cut z=0.5x+3 (off-axis, not the equator)
          create sphere 10 as @s
          trim @s by plane -0.5 0 1 3
          check validity @s

          // 5) NURBS dome cut at x=0 -- keep x>0; 'keep below' keeps the other half
          create nurbs 20 8 as @n
          trim @n by plane 1 0 0 0
          create nurbs 20 8 as @m
          trim @m by plane 1 0 0 0 keep below

          // 6) NURBS dome CAP cut z>3 -- the surface crosses, the flat boundary
          //    does not; the curved cap is kept and faceted along the section
          create nurbs 20 8 as @cap
          trim @cap by plane 0 0 1 3
          disp math $face

          // 7) Crop a NURBS dome to its central quarter (keep u,v in [0.25,0.75])
          create nurbs 20 8 as @d
          trim surface $face keep 0.25 0.75 0.25 0.75
          disp math $face

          // 8) Split a plane's boundary edge at its midpoint (keeps both halves)
          create plane 10 10 as @c
          trim curve $edge at 0.5
        """)

    def help_extend(self) -> None:
        self._h(f"""
        extend -- grow a SOURCE entity until it reaches a TARGET entity
        {self._HR}
        Reuses the same geometric-intersection math as 'trim' (line-plane,
        ray-surface). It updates BOTH the geometry (a new point placed exactly on
        the target) AND the topology (a new vertex/edge, or a swept strip of
        faces). The source is extended TO the target.

        Forms:

        1)  extend #<edge> to plane <nx> <ny> <nz> <d>
        2)  extend #<edge> to #<face>
            Extend a CURVE/segment. A ray is cast from the edge end along its
            tangent and met with the target; a new vertex is inserted at the
            exact contact point and a new edge is appended (Euler +1V +1E). The
            end whose forward tangent reaches the target is chosen automatically.
              target 'plane'   -> line-plane intersection (closed form)
              target #<face>   -> plane of that face, or its NURBS surface when
                                  it carries one: a tessellation hit seeds a
                                  Newton iteration (closest-point projection on
                                  the surface tangents), so the contact lies on
                                  the TRUE surface and stores its (u,v) address
                                  (shown by 'disp math #<vertex>')

        3)  extend #<face> to plane <nx> <ny> <nz> <d> [along <dx> <dy> <dz>]
        4)  extend #<face> to #<face> [along <dx> <dy> <dz>]
            Extend a PLANAR sheet by sweeping it up to the target: every boundary
            vertex travels along the sweep direction until it meets the target,
            and side faces are spun out (MEF), exactly like 'extrude' but stopping
            ON the target. Against a plane the cap is flat; against a NURBS surface
            each vertex stops on the surface, so the cap conforms to it.
              along <dx dy dz>  optional sweep direction; default is the face
                                normal, auto-flipped toward the target.

        Combinations (source - target):
          curve - plane   form 1     curve - nurbs   form 2 (target NURBS face)
          plane - plane   form 3     plane - nurbs   form 4 (target NURBS face)

        Arrange source and target so the extension path actually crosses the
        target (a right angle or an oblique angle) -- a parallel target is never
        reached, or gives only a trivial offset.

        Examples (each block is self-contained -- copy/paste and run as-is):

          // 1) curve -> plane : a wire grows +x into the perpendicular wall x=20
          micro mvfs 0 0 0 as @w
          micro mev $vertex 6 0 0
          extend $edge to plane 1 0 0 20        // hits the wall head-on at (20,0,0)

          // 2) curve -> nurbs : a wire below a dome rises +z onto its surface
          create nurbs 20 8 as @dome
          set @df $face                          // capture the dome's NURBS face
          micro mvfs 0 0 -6 as @wire
          micro mev $vertex 0 0 -3
          extend $edge to @df                    // lands on the dome apex (0,0,4)

          // 3) plane -> plane : sweep a sheet onto a SLANTED plane (a ramp)
          create plane 20 20 as @s
          extend $face to plane -0.5 0 1 10      // not parallel -> cap is a ramp

          // 4) plane -> nurbs : grow a sheet up onto a dome (cap conforms)
          create nurbs 20 8 as @dome
          set @df $face
          create plane 12 12 as @sheet
          move @sheet 0 0 -3
          extend $face to @df
        """)

    def help_intersect(self) -> None:
        self._h(f"""
        intersect -- NURBS x NURBS surface-surface intersection (SSI)
        {self._HR}
        intersect #<faceA> #<faceB> [samples <n>] [as @name]

        Computes the intersection curve of two NURBS faces and lifts it into
        the model as a WIRE solid (vertex/edge chain; a closed loop is closed
        with MEF). The algorithm turns surface B into a signed-distance field
        over surface A's (u,v) grid (closest-point projection + normal sign),
        marches the zero set in A's parameter space, refines every point by
        bisection on the true surface, then tightens each point onto the exact
        intersection by alternating projections between BOTH surfaces.

        The curve is delivered in all three classic representations:
          * the 3D wire itself (view / save / disp)
          * (u,v) on surface A   -> vertex.on_surface_uv    (disp math #<vertex>)
          * (u,v) on surface B   -> vertex.on_surface_uv_b
        Branch data is also recorded on both faces (disp math #<face>).

        Example (two domes arranged to cross -- copy/paste and run as-is):
          create nurbs 20 8 as @a
          set @fa $face
          create nurbs 20 8 as @b
          set @fb $face
          rotate @b x 180                  // flip the second dome into a bowl
          move @b 0 0 5                    // the bowl now dips into the dome
          intersect @fa @fb as @c          // -> closed intersection loop
          check validity @c
          disp math $vertex                // (u,v) on BOTH surfaces
        """)

    def help_blend(self) -> None:
        self._h(f"""
        blend -- curvature-continuous (G2) blend patch across an intersection
        {self._HR}
        blend #<faceA> #<faceB> width <w> [samples <n>] [as @name]

        Builds a smooth NURBS strip joining surface A to surface B across
        their intersection curve:

          1. The NURBS x NURBS intersection is computed (see 'help intersect').
          2. On each surface a RAIL is offset 'width' from the curve along the
             in-surface direction perpendicular to it (away from the other
             surface, orientation kept consistent along the curve).
          3. At each rail the walk toward the intersection is differentiated
             on the true surface (1st + 2nd directional derivatives).
          4. Each cross-section is a QUINTIC HERMITE span matching position,
             first AND second derivative at both rails -- a curvature-
             continuous join with each host surface.
          5. Control rows are solved so the patch INTERPOLATES every sampled
             section (basis-matrix solve), then attached to a lamina solid.

        Example (self-contained -- copy/paste and run as-is):
          create nurbs 20 8 as @a
          set @fa $face
          create nurbs 20 8 as @b
          set @fb $face
          rotate @b x 180
          move @b 0 0 5
          blend @fa @fb width 1.5 as @bl   // degree 5 x 3 Hermite patch
          check validity @bl
          view @bl                         // rendered via exact B-spline eval
        """)

    def help_delete(self) -> None:
        self._h(f"""
        delete -- remove a solid from the model
        {self._HR}
        delete #<solid>|@alias|$solid

        Unregisters the solid and every vertex/edge/half-edge/loop/face it
        owns. Aliases that pointed into the deleted solid are unbound.
        """)

    def help_micro(self) -> None:
        self._h(f"""
        micro <op> <args>  -- atomic Euler topology operators
        {self._HR}
        micro mvfs <x> <y> <z> [as #id|@name]
            Make Vertex Face Solid -- seed a new solid at (x, y, z).
            Returns: solid, face, vertex.
            ΔEuler: +1V +1F  (starts genus=0, no edges)

        micro mev #<vertex> <x> <y> <z> [as #edge|@name]
            Make Edge Vertex -- grow a wire by adding vertex at (x,y,z)
            connected to #<vertex>.
            ΔEuler: +1V +1E

        micro mef #<v1> #<v2> [as #face|@name]
            Make Edge Face -- close an open loop between two existing
            vertices v1 and v2, splitting it into a new face.
            ΔEuler: +1E +1F

        micro kev #<edge>
            Kill Edge Vertex (inverse of mev) -- remove a dangling spike edge
            together with its tip vertex. A one-edge wire degenerates back to
            the mvfs seed state.
            ΔEuler: -1V -1E

        micro kef #<edge>
            Kill Edge Face (inverse of mef) -- remove an edge separating two
            distinct faces, merging them into one.
            ΔEuler: -1E -1F

        micro semv #<edge> <x> <y> <z>          (alias: micro split)
            Split Edge Make Vertex -- insert a vertex at (x,y,z) on the edge,
            updating both adjacent loops coherently.
            ΔEuler: +1V +1E

        micro kemr #<edge>
            Kill Edge Make Ring -- remove an edge bordered twice by the same
            face; the detached cycle becomes an inner ring of that face.
            ΔEuler: -1E +1R

        micro mekr #<v_outer> #<v_ring>
            Make Edge Kill Ring (inverse of kemr) -- bridge an inner ring back
            to the outer loop of its face with a new edge.
            ΔEuler: +1E -1R

        micro kfmr #<face>
            Kill Face Make Ring -- destroy a single-loop face and re-home its
            loop as an inner ring of the adjacent face (opens a handle).
            ΔEuler: -1F +1R, genus+1

        micro mfkr #<loop>
            Make Face Kill Ring (inverse of kfmr) -- promote an inner ring
            loop into its own face (closes a handle).
            ΔEuler: +1F -1R, genus-1

        Manual box example (= 'create box 10 10 5'):
          micro mvfs 0 0 0
          micro mev $vertex 10 0 0
          micro mev $vertex 10 10 0
          micro mev $vertex 0 10 0
          micro mef $vertex #<first_v>    // or use @alias to track first vertex
          extrude $face 0 0 5

        Spike + undo example:
          micro mvfs 0 0 0 as @w
          micro mev $vertex 5 0 0         // +1V +1E
          micro kev $edge                 // -1V -1E: back to the seed
        """)

    # ── geometry / editing ───────────────────────────────────────────── #

    def help_setpoint(self) -> None:
        self._h(f"""
        setpoint #<vertex> as (<x>, <y>, <z>)
        {self._HR}
        Update the 3-D coordinates of a vertex.
          #<vertex>      target vertex (use $vertex, @alias, or #id)
          (<x>, <y>, <z>)  new position  (parentheses and commas optional)

        Examples:
          setpoint $vertex as (1, 2, 3)
          setpoint #105 as 5 0 0
        """)

    def help_setcurve(self) -> None:
        self._h(f"""
        setcurve #<edge> as bezier #<cp1> #<cp2> ... #<cpN>
        {self._HR}
        Attach a Bezier curve to an edge using existing vertices as
        control points.  Degree = N-1  (min 2 control points → degree 1).
          #<edge>       target edge
          #<cpI>        vertex ids used as control points (≥ 2 required)

        Example:
          create box 10 10 10 as @b
          setcurve $edge as bezier $vertex #103 #105
        """)

    def help_setsurface(self) -> None:
        self._h(f"""
        setsurface #<face> as nurbs <deg_u> <deg_v>
        {self._HR}
        Attach a NURBS patch to a face, using the face's own boundary
        vertices as control points.
          #<face>    target face
          <deg_u>    degree in the U parameter direction
          <deg_v>    degree in the V parameter direction

        The face must have ≥ (deg_u+1)×(deg_v+1) boundary vertices;
        otherwise a bilinear (degree-1) fallback patch is used.

        Example:
          create plane 10 10 as @p
          setsurface $face as nurbs 2 2
          disp math $face
        """)

    # ── transforms ───────────────────────────────────────────────────── #

    def help_move(self) -> None:
        self._h(f"""
        move #<entity> <dx> <dy> <dz>
        {self._HR}
        Translate an entity (solid, face, edge, or vertex) by (dx, dy, dz).
        All vertices of the entity are shifted by the same vector.

        Examples:
          move @box 5 0 0          // shift the aliased solid 5 units in X
          move $solid 0 0 -10      // move last solid 10 units down
          move #105 1 1 1          // move a single vertex
        """)

    def help_rotate(self) -> None:
        self._h(f"""
        rotate #<entity> <x|y|z> <angle_deg>
        {self._HR}
        Rotate an entity about a principal axis through the origin.
          <x|y|z>      rotation axis
          <angle_deg>  rotation angle in degrees (positive = CCW when
                       looking from positive axis toward origin)

        Examples:
          rotate @box z 45         // rotate 45° about Z
          rotate $solid x 90       // rotate last solid 90° about X
          rotate #100 y -30        // rotate solid #100 by -30° about Y
        """)

    def help_scale(self) -> None:
        self._h(f"""
        scale #<entity> <factor>
        {self._HR}
        Uniformly scale an entity about its centroid.
          <factor>  scale multiplier (e.g. 2.0 doubles all dimensions,
                    0.5 halves them)

        Examples:
          scale @box 2             // double the box size
          scale $solid 0.1         // shrink last solid to 10%
          scale #100 1.5           // scale solid #100 by 1.5×
        """)

    # ── query & I/O ──────────────────────────────────────────────────── #

    def help_disp(self) -> None:
        self._h(f"""
        disp topology|vertices|math [#id]
        {self._HR}
        disp topology [#solid]
            Print the full half-edge tree: faces → loops → half-edges,
            showing vertex ids, edge ids, next/prev/twin links.

        disp vertices [#solid]
            Print a coordinate table for every vertex of the solid.

        disp math #<id>
            Show the underlying geometry of one entity:
              vertex  → 3-D point coordinates
              edge    → straight segment or Bezier control points
              face    → NURBS control net, trim plane, trim section

        Examples:
          disp topology @box       // half-edge tree for the aliased solid
          disp vertices $solid     // coordinates of last solid
          disp math $face          // geometry of last face
          disp math #107           // geometry of entity #107
        """)

    def help_check(self) -> None:
        self._h(f"""
        check validity [#solid]
        {self._HR}
        Run topology integrity checks on one or all solids:
          • Euler–Poincaré:  V - E + F - R = 2(S - G)
          • Half-edge pointers: every next/prev/twin link is consistent

        Reports PASS or FAIL with a breakdown of V, E, F, rings, shells, genus.

        Examples:
          check validity            // check all solids
          check validity @box       // check specific solid
          check validity $solid     // check last created solid
        """)

    def help_save(self) -> None:
        self._h(f"""
        save "<file.step>" [#solid] [faceted]
        {self._HR}
        Export a solid to STEP AP203 format.
          "<file.step>"  output file path (quotes required if path has spaces)
          [#solid]       optional solid to save; defaults to the first solid
          [faceted]      export trimmed NURBS faces as keep-side triangle
                         shells instead of analytic trimmed B-splines (for
                         viewers with weak pcurve support)

        For untrimmed solids:   writes MANIFOLD_SOLID_BREP + CLOSED_SHELL
        For trimmed laminas:    writes SHELL_BASED_SURFACE_MODEL + OPEN_SHELL
                                (discarded half-faces are omitted)
        NURBS faces export as B_SPLINE_SURFACE_WITH_KNOTS. A TRIMMED NURBS
        face exports analytically by default: the full B-spline surface
        bounded by its topological loops, each edge a SURFACE_CURVE pairing
        the 3D chord with a PCURVE (a 2D B-spline in the surface's (u,v)
        parameter space inside a DEFINITIONAL_REPRESENTATION, ISO 10303-42).

        Examples:
          save "box.step"                 // save first solid
          save "out/nurbs.step" @dome     // save aliased solid
          save "scripts/out.step" $solid  // save last solid

        Trim + save workflow:
          create nurbs 20 8 as @n
          trim @n by plane 1 0 0 0
          save "half_dome.step" @n        // OPEN_SHELL with reparameterized NURBS
        """)

    def help_load(self) -> None:
        self._h(f"""
        load "<file.step>"
        {self._HR}
        Import CARTESIAN_POINT / VERTEX_POINT entities from a STEP file
        into a new solid as a vertex cloud.

        Note: full half-edge reconstruction from STEP is not supported;
        only the point geometry is recovered.  The number of recovered
        points is reported.

        Example:
          load "box.step"
          disp vertices $solid
        """)

    def help_run(self) -> None:
        self._h(f"""
        run "<script_file>"
        {self._HR}
        Execute commands from a text file, line by line.
        Blank lines and lines starting with '#' or '//' are skipped.
        '//' also starts an inline comment.

        Examples:
          run "scripts/box.bcmd"
          run "scripts/trim_examples.bcmd"
        """)

    def help_view(self) -> None:
        self._h(f"""
        view [solid|wire|points] [#id|@alias|$solid]
        {self._HR}
        Open an interactive 3-D viewer for a solid.

        Mode (optional, default: solid):
          solid   / shaded     shaded faces + edge overlay
          wire    / wireframe  edge wireframe only
          points  / pt         vertex cloud only

        Target (optional, default: last created solid):
          @alias, $solid, #id, or a literal entity id

        Backend priority (first available wins):
          1. plotly  -- browser-based HTML viewer, interactive rotate/zoom/pan
                        no numpy required  (pip install plotly)
          2. vedo    -- VTK window, requires working numpy  (pip install vedo)
          3. tkinter -- stdlib wireframe fallback, always available

        NURBS faces are shown as a sampled surface mesh in plotly and vedo,
        and as a grid of sample lines in the tkinter fallback.

        Examples:
          view                        // view last solid, shaded mode
          view @box                   // view aliased solid
          view wire $solid            // wireframe of last solid
          view solid @dome            // shaded view of @dome
          view points #100            // vertex cloud of solid #100

          create nurbs 20 8 as @n
          trim @n by plane 1 0 0 0
          view @n                     // shows trimmed NURBS (keep half only)
        """)

    # ------------------------------------------------------------------ #
    # Small parsing utilities
    # ------------------------------------------------------------------ #
    def _split_on_keyword(self, arg: str, keyword: str) -> tuple[List[str], List[str]]:
        """Split shlex tokens into (before, after) around a keyword like 'as'."""
        tokens = shlex.split(arg)
        kw = keyword.lower()
        for i, t in enumerate(tokens):
            if t.lower() == kw:
                return tokens[:i], tokens[i + 1:]
        raise CliError(f"missing '{keyword}' keyword")

    @staticmethod
    def _parse_coords(tokens: List[str]) -> List[float]:
        """Parse '(1, 2, 3)' style tokens into a list of floats."""
        joined = " ".join(tokens).replace("(", " ").replace(")", " ").replace(",", " ")
        return [float(t) for t in joined.split()]
