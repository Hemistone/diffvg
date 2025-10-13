"""Scene writing mixin used by OptimizableSvg."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as etree
from xml.dom import minidom


class SvgWriterMixin:
    def write_xml(self) -> str:
        tree = self.root.write_xml(self)
        return minidom.parseString(etree.tostring(tree, "utf-8")).toprettyxml(indent="  ")

    def write_defs(self, root: etree.Element) -> None:
        if len(self.defs) == 0:
            return

        defnode = etree.SubElement(root, "defs")
        stylenode = etree.SubElement(root, "style")
        stylenode.set("type", "text/css")
        stylenode.text = ""

        defcpy = copy.copy(self.defs)
        while len(defcpy) > 0:
            to_remove = []
            for key, value in defcpy.items():
                if issubclass(value.__class__, self.__class__.SvgNode):
                    if value.href is None or value.href not in defcpy:
                        value.write_xml(defnode)
                        to_remove.append(key)
                else:
                    stylenode.text += key + " {" + value + "}\n"
                    to_remove.append(key)

            for key in to_remove:
                del defcpy[key]


__all__ = ["SvgWriterMixin"]
