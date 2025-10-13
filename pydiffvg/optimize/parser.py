"""SVG parsing mixin used by OptimizableSvg."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import cssutils
import numpy as np
import torch
import xml.etree.ElementTree as etree

import pydiffvg


class SvgParserMixin:
    unit_dict = {
        "px": 1,
        "mm": 4,
        "cm": 40,
        "in": 25.4 * 4,
        "pt": 25.4 * 4 / 72,
        "pc": 25.4 * 4 / 6,
    }
    appearance_keys = [
        "fill",
        "fill-opacity",
        "fill-rule",
        "opacity",
        "stroke",
        "stroke-opacity",
        "stroke-width",
    ]

    @staticmethod
    def remove_namespace(s: str) -> str:
        """
            {...} ... -> ...
        """
        return re.sub("{.*}", "", s)

    @staticmethod
    def is_namespace(s: str) -> bool:
        return re.match("{.*}", s) is not None

    @classmethod
    def parseTransform(cls, node: etree.Element):
        if "transform" not in node.attrib and "gradientTransform" not in node.attrib:
            return None

        tf_string = node.attrib["transform"] if "transform" in node.attrib else node.attrib["gradientTransform"]
        tforms = tf_string.split(")")[:-1]
        mat = np.eye(3)
        for tform in tforms:
            ttype = tform.split("(")[0]
            args = [float(val) for val in re.split("[, ]+", tform.split("(")[1])]
            if ttype == "matrix":
                mat = mat @ cls.TransformTools.parse_matrix(args)
            elif ttype == "translate":
                mat = mat @ cls.TransformTools.parse_translate(args)
            elif ttype == "rotate":
                mat = mat @ cls.TransformTools.parse_rotate(args)
            elif ttype == "scale":
                mat = mat @ cls.TransformTools.parse_scale(args)
            elif ttype == "skewX":
                mat = mat @ cls.TransformTools.parse_skewx(args)
            elif ttype == "skewY":
                mat = mat @ cls.TransformTools.parse_skewy(args)
            else:
                raise ValueError(f"Unknown transform type '{ttype}'")
        return mat

    @classmethod
    def parseLength(cls, s: str) -> float:
        val = None
        unit = ""
        for i in range(len(s)):
            try:
                val = float(s[: len(s) - i])
                unit = s[len(s) - i :]
                break
            except ValueError:
                continue
        if val is None:
            raise ValueError(f"Unable to parse length string '{s}'")
        if len(unit) > 0 and unit not in cls.unit_dict:
            raise ValueError(f"Unknown or unsupported unit '{unit}' encountered while parsing")
        if unit != "":
            val *= cls.unit_dict[unit]
        return val

    @staticmethod
    def parseOpacity(s: str) -> float:
        is_percent = s.endswith("%")
        s = s.rstrip("%")
        val = float(s)
        if is_percent:
            val = val / 100
        return float(np.clip(val, 0.0, 1.0))

    @staticmethod
    def parse_color(s: str) -> torch.Tensor:
        if not s.startswith("#"):
            raise ValueError(f"Color argument `{s}` not supported")
        s = s.lstrip("#")
        if len(s) == 6:
            rgb = tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
            return torch.tensor([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0])
        if len(s) == 3:
            rgb = tuple((int(s[i : i + 1], 16)) for i in (0, 1, 2))
            return torch.tensor([rgb[0] / 15.0, rgb[1] / 15.0, rgb[2] / 15.0])
        raise ValueError(f"Color argument `{s}` not supported")

    @staticmethod
    def rgb_to_string(val: torch.Tensor) -> str:
        byte_rgb = (val.clone().detach() * 255).type(torch.int)
        byte_rgb.clamp_(min=0, max=255)
        return "#{:02x}{:02x}{:02x}".format(*byte_rgb)

    @classmethod
    def parsePaint(cls, paintStr: str, defs: Dict[str, any], device: torch.device):
        paintStr = paintStr.strip()
        if paintStr == "none":
            return ("none", None)
        if paintStr.startswith("#"):
            return ("solid", cls.parse_color(paintStr).to(device))
        if paintStr.startswith("url"):
            url = paintStr.lstrip("url(").rstrip(")").strip("'\"").lstrip("#")
            if url not in defs:
                raise ValueError(f"Paint-type attribute referencing an unknown object with ID '#{url}'")
            return ("url", defs[url])
        raise ValueError(f"Unrecognized paint string: '{paintStr}'")

    @classmethod
    def parseAppearance(cls, node: etree.Element, defs, device: torch.device):
        ret = {}
        parse_keys = cls.appearance_keys
        local_dict = {key: value for key, value in node.attrib.items() if key in parse_keys}
        css_dict = {}
        style_dict = {}
        appearance_dict = {}
        if "class" in node.attrib:
            cls_name = node.attrib["class"]
            if "." + cls_name in defs:
                css_string = defs["." + cls_name]
                css_dict = {
                    item.split(":")[0]: item.split(":")[1]
                    for item in css_string.split(";")
                    if len(item) > 0 and item.split(":")[0] in parse_keys
                }
        if "style" in node.attrib:
            style_string = node.attrib["style"]
            style_dict = {
                item.split(":")[0]: item.split(":")[1]
                for item in style_string.split(";")
                if len(item) > 0 and item.split(":")[0] in parse_keys
            }
        appearance_dict.update(css_dict)
        appearance_dict.update(style_dict)
        appearance_dict.update(local_dict)
        for key, value in appearance_dict.items():
            if key == "fill":
                ret[key] = cls.parsePaint(value, defs, device)
            elif key == "fill-opacity":
                ret[key] = torch.tensor(cls.parseOpacity(value), device=device)
            elif key == "fill-rule":
                ret[key] = value
            elif key == "opacity":
                ret[key] = torch.tensor(cls.parseOpacity(value), device=device)
            elif key == "stroke":
                ret[key] = cls.parsePaint(value, defs, device)
            elif key == "stroke-opacity":
                ret[key] = torch.tensor(cls.parseOpacity(value), device=device)
            elif key == "stroke-width":
                ret[key] = torch.tensor(cls.parseLength(value), device=device)
            else:
                raise ValueError(
                    f"Error while parsing appearance attributes: key '{key}' should not be here"
                )

        return ret

    def parseRoot(self, root: etree.Element):
        cls = self.__class__
        if self.verbose:
            print(self.offset_str("Parsing root"))
        self.depth += 1

        self.parseViewport(root)
        canvmax = np.max(self.canvas)
        self.settings.global_override(["transforms", "translation_mult"], canvmax)
        node_id = root.attrib["id"] if "id" in root.attrib else None

        transform = cls.parseTransform(root)
        appearance = cls.parseAppearance(root, self.defs, self.device)

        version = root.attrib["version"] if "version" in root.attrib else "<unknown version>"
        if version != "2.0":
            print(
                self.offset_str(
                    f"Warning: Version {version} is not 2.0, strange things may happen"
                )
            )

        self.root = cls.RootNode(node_id, transform, appearance, self.settings)

        if self.verbose:
            self.reportSkippedAttribs(
                root,
                ["width", "height", "id", "transform", "version", "style"] + cls.appearance_keys,
            )

        skipped = []
        for child in root:
            tag = cls.remove_namespace(child.tag)
            if tag in cls.recognised_shapes:
                self.parseShape(child, self.root)
            elif tag == "defs":
                self.parseDefs(child)
            elif tag == "style":
                self.parseStyle(child)
            elif tag == "g":
                self.parseGroup(child, self.root)
            else:
                skipped.append(child)

        if self.verbose:
            self.reportSkippedChildren(root, skipped)

        self.depth -= 1

    def parseShape(self, shape: etree.Element, parent):
        cls = self.__class__
        tag = cls.remove_namespace(shape.tag)
        if self.verbose:
            print(
                self.offset_str(
                    "Parsing {}#{}".format(
                        tag, shape.attrib["id"] if "id" in shape.attrib else "<No ID>"
                    )
                )
            )

        self.depth += 1
        if tag == "path":
            self.parsePath(shape, parent)
        elif tag == "circle":
            self.parseCircle(shape, parent)
        elif tag == "rect":
            self.parseRect(shape, parent)
        elif tag == "ellipse":
            self.parseEllipse(shape, parent)
        elif tag == "polygon":
            self.parsePolygon(shape, parent)
        else:
            raise ValueError(f"Encountered unknown shape type '{tag}'")
        self.depth -= 1

    def parsePath(self, shape: etree.Element, parent):
        cls = self.__class__
        path_string = shape.attrib["d"]
        name = shape.attrib["id"] if "id" in shape.attrib else ""
        paths = pydiffvg.from_svg_path(path_string)
        for idx, path in enumerate(paths):
            path.stroke_width = torch.tensor([0.0], device=self.device)
            path.num_control_points = path.num_control_points.to(self.device)
            path.points = path.points.to(self.device)
            path.source_id = name
            path.id = f"{name}-{idx}" if len(paths) > 1 else name
        transform = cls.parseTransform(shape)
        appearance = cls.parseAppearance(shape, self.defs, self.device)
        node = cls.PathNode(name, transform, appearance, self.settings, paths)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(
                shape, ["id", "d", "transform", "style"] + cls.appearance_keys
            )
            self.reportSkippedChildren(shape, list(shape))

    def parseEllipse(self, shape: etree.Element, parent):
        cls = self.__class__
        cx = float(shape.attrib["cx"]) if "cx" in shape.attrib else 0.0
        cy = float(shape.attrib["cy"]) if "cy" in shape.attrib else 0.0
        rx = float(shape.attrib["rx"])
        ry = float(shape.attrib["ry"])
        name = shape.attrib["id"] if "id" in shape.attrib else "<No ID>"
        transform = cls.parseTransform(shape)
        appearance = cls.parseAppearance(shape, self.defs, self.device)
        node = cls.EllipseNode(name, transform, appearance, self.settings, (cx, cy, rx, ry))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(
                shape,
                ["id", "x", "y", "r", "transform", "style"] + cls.appearance_keys,
            )
            self.reportSkippedChildren(shape, list(shape))

    def parsePolygon(self, shape: etree.Element, parent):
        cls = self.__class__
        points_string = shape.attrib["points"]
        points = []
        for point_string in points_string.split(" "):
            if len(point_string) == 0:
                continue
            coord_strings = point_string.split(",")
            assert len(coord_strings) == 2
            points.append([float(coord_strings[0]), float(coord_strings[1])])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
        name = shape.attrib["id"] if "id" in shape.attrib else "<No ID>"
        transform = cls.parseTransform(shape)
        appearance = cls.parseAppearance(shape, self.defs, self.device)
        node = cls.PolygonNode(name, transform, appearance, self.settings, points_tensor)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(
                shape,
                ["id", "points", "transform", "style"] + cls.appearance_keys,
            )
            self.reportSkippedChildren(shape, list(shape))

    def parseCircle(self, shape: etree.Element, parent):
        cls = self.__class__
        cx = float(shape.attrib["cx"]) if "cx" in shape.attrib else 0.0
        cy = float(shape.attrib["cy"]) if "cy" in shape.attrib else 0.0
        r = float(shape.attrib["r"])
        name = shape.attrib["id"] if "id" in shape.attrib else "<No ID>"
        transform = cls.parseTransform(shape)
        appearance = cls.parseAppearance(shape, self.defs, self.device)
        node = cls.CircleNode(name, transform, appearance, self.settings, (cx, cy, r))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(
                shape, ["id", "x", "y", "r", "transform", "style"] + cls.appearance_keys
            )
            self.reportSkippedChildren(shape, list(shape))

    def parseRect(self, shape: etree.Element, parent):
        cls = self.__class__
        x = float(shape.attrib["x"]) if "x" in shape.attrib else 0.0
        y = float(shape.attrib["y"]) if "y" in shape.attrib else 0.0
        width = float(shape.attrib["width"])
        height = float(shape.attrib["height"])
        name = shape.attrib["id"] if "id" in shape.attrib else "<No ID>"
        transform = cls.parseTransform(shape)
        appearance = cls.parseAppearance(shape, self.defs, self.device)
        node = cls.RectNode(name, transform, appearance, self.settings, (x, y, width, height))
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(
                shape,
                ["id", "x", "y", "width", "height", "transform", "style"] + cls.appearance_keys,
            )
            self.reportSkippedChildren(shape, list(shape))

    def parseGroup(self, group: etree.Element, parent):
        cls = self.__class__
        tag = cls.remove_namespace(group.tag)
        node_id = group.attrib["id"] if "id" in group.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, node_id)))

        self.depth += 1

        transform = self.parseTransform(group)
        appearance = cls.parseAppearance(group, self.defs, self.device)
        node = cls.GroupNode(node_id, transform, appearance, self.settings)
        parent.children.append(node)

        if self.verbose:
            self.reportSkippedAttribs(
                group, ["id", "transform", "style"] + cls.appearance_keys
            )

        skipped_children = []
        for child in group:
            child_tag = cls.remove_namespace(child.tag)
            if child_tag in cls.recognised_shapes:
                self.parseShape(child, node)
            elif child_tag == "defs":
                self.parseDefs(child)
            elif child_tag == "style":
                self.parseStyle(child)
            elif child_tag == "g":
                self.parseGroup(child, node)
            else:
                skipped_children.append(child)

        if self.verbose:
            self.reportSkippedChildren(group, skipped_children)

        self.depth -= 1

    def parseStyle(self, style_node: etree.Element):
        cls = self.__class__
        tag = cls.remove_namespace(style_node.tag)
        node_id = style_node.attrib["id"] if "id" in style_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, node_id)))

        if style_node.attrib.get("type", "text/css") != "text/css":
            raise ValueError(
                f"Only text/css style recognized, got {style_node.attrib.get('type')}"
            )

        self.depth += 1

        node = cls.SvgNode(node_id, None, {}, self.settings)

        if self.verbose:
            self.reportSkippedAttribs(style_node, ["id", "type"])

        if len(style_node) > 0:
            raise ValueError("Style node should not have children (has {})".format(len(style_node)))

        sheet = cssutils.parseString(style_node.text)
        for rule in sheet:
            if hasattr(rule, "selectorText") and hasattr(rule, "style"):
                name = rule.selectorText
                if len(name) >= 2 and name[0] == ".":
                    self.defs[name] = rule.style.getCssText().replace("\n", "")
                else:
                    raise ValueError(f"Unrecognized CSS selector {name}")
            else:
                raise ValueError("No style or selector text in CSS rule")

        self.depth -= 1

    def parseDefs(self, def_node: etree.Element):
        cls = self.__class__
        tag = cls.remove_namespace(def_node.tag)
        node_id = def_node.attrib["id"] if "id" in def_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, node_id)))

        self.depth += 1

        node = cls.SvgNode(node_id, None, {}, self.settings)

        if self.verbose:
            self.reportSkippedAttribs(def_node, ["id"])

        skipped_children = []
        for child in def_node:
            child_tag = cls.remove_namespace(child.tag)
            if child_tag == "linearGradient":
                self.parseGradient(child, node)
            elif child_tag in cls.recognised_shapes:
                raise NotImplementedError("Definition/instantiation of shapes not supported")
            elif child_tag == "defs":
                raise NotImplementedError("Definition within definition not supported")
            elif child_tag == "g":
                raise NotImplementedError("Groups within definition not supported")
            else:
                skipped_children.append(child)

            if len(node.children) > 0:
                self.defs[node.children[0].id] = node.children[0]
                node.children.pop()

        if self.verbose:
            self.reportSkippedChildren(def_node, skipped_children)

        self.depth -= 1

    def parseGradientStop(self, stop: etree.Element) -> Tuple[float, torch.Tensor, float]:
        cls = self.__class__
        param_dict = {
            key: value
            for key, value in stop.attrib.items()
            if key in ["id", "offset", "stop-color", "stop-opacity"]
        }
        if "style" in stop.attrib:
            style_dict = {
                item.split(":")[0]: item.split(":")[1]
                for item in stop.attrib["style"].split(";")
                if len(item) > 0
            }
            param_dict.update(style_dict)

        offset = cls.parseOpacity(param_dict["offset"])
        color = cls.parse_color(param_dict["stop-color"])
        opacity = (
            cls.parseOpacity(param_dict["stop-opacity"])
            if "stop-opacity" in param_dict
            else 1.0
        )

        return offset, color, opacity

    def parseGradient(self, gradient_node: etree.Element, parent):
        cls = self.__class__
        tag = cls.remove_namespace(gradient_node.tag)
        node_id = gradient_node.attrib["id"] if "id" in gradient_node.attrib else "<No ID>"
        if self.verbose:
            print(self.offset_str("Parsing {}#{}".format(tag, node_id)))

        self.depth += 1
        if "stop" not in [cls.remove_namespace(child.tag) for child in gradient_node] and "href" not in [
            cls.remove_namespace(key) for key in gradient_node.attrib.keys()
        ]:
            raise ValueError(f"Gradient {node_id} has neither stops nor a href link to them")

        transform = self.parseTransform(gradient_node)
        begin = None
        end = None
        offsets: List[float] = []
        stops: List[torch.Tensor] = []
        href = None

        if "x1" in gradient_node.attrib or "y1" in gradient_node.attrib:
            begin = np.array([0.0, 0.0])
            if "x1" in gradient_node.attrib:
                begin[0] = float(gradient_node.attrib["x1"])
            if "y1" in gradient_node.attrib:
                begin[1] = float(gradient_node.attrib["y1"])
            begin = torch.tensor(begin.transpose(), dtype=torch.float32)

        if "x2" in gradient_node.attrib or "y2" in gradient_node.attrib:
            end = np.array([0.0, 0.0])
            if "x2" in gradient_node.attrib:
                end[0] = float(gradient_node.attrib["x2"])
            if "y2" in gradient_node.attrib:
                end[1] = float(gradient_node.attrib["y2"])
            end = torch.tensor(end.transpose(), dtype=torch.float32)

        stop_nodes = [
            node for node in list(gradient_node) if cls.remove_namespace(node.tag) == "stop"
        ]
        if len(stop_nodes) > 0:
            stop_nodes = sorted(stop_nodes, key=lambda n: float(n.attrib["offset"]))

            for stop in stop_nodes:
                offset, color, opacity = self.parseGradientStop(stop)
                offsets.append(offset)
                stops.append(torch.cat((color, torch.tensor([opacity]))))

        hkey = next(
            (
                value
                for key, value in gradient_node.attrib.items()
                if cls.remove_namespace(key) == "href"
            ),
            None,
        )
        if hkey is not None:
            href = self.defs[hkey.lstrip("#")]

        parent.children.append(
            cls.GradientNode(
                node_id,
                transform,
                self.settings,
                begin.to(self.device) if begin is not None else begin,
                end.to(self.device) if end is not None else end,
                torch.tensor(offsets, dtype=torch.float32, device=self.device)
                if len(offsets) > 0
                else None,
                torch.stack(stops).to(self.device) if len(stops) > 0 else None,
                href,
            )
        )

        self.depth -= 1

    def parseViewport(self, root: etree.Element):
        if "width" in root.attrib and "height" in root.attrib:
            self.canvas = np.array(
                [
                    int(math.ceil(float(root.attrib["width"]))),
                    int(math.ceil(float(root.attrib["height"]))),
                ]
            )
        elif "viewBox" in root.attrib:
            s = root.attrib["viewBox"].split(" ")
            w = s[2]
            h = s[3]
            self.canvas = np.array(
                [int(math.ceil(float(w))), int(math.ceil(float(h)))]
            )
        else:
            raise ValueError("Size information is missing from document definition")


__all__ = ["SvgParserMixin"]
