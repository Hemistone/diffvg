"""Scene graph node classes for SVG optimization."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as etree

import numpy as np
import torch

import pydiffvg

from .settings import SvgOptimizationSettings

_TRANSFORM_TOOLS = None
_TRANSFORM_OPTIMIZER = None
_COLOR_OPTIMIZER = None
_GRADIENT_OPTIMIZER = None
_RGB_TO_STRING = None


def configure_scene_graph(*, transform_tools, transform_optimizer, color_optimizer, gradient_optimizer, rgb_to_string) -> None:
    global _TRANSFORM_TOOLS, _TRANSFORM_OPTIMIZER, _COLOR_OPTIMIZER, _GRADIENT_OPTIMIZER, _RGB_TO_STRING
    _TRANSFORM_TOOLS = transform_tools
    _TRANSFORM_OPTIMIZER = transform_optimizer
    _COLOR_OPTIMIZER = color_optimizer
    _GRADIENT_OPTIMIZER = gradient_optimizer
    _RGB_TO_STRING = rgb_to_string


class SvgNode:
    def __init__(self,id,transform,appearance,settings):
        self.id=id
        self.children=[]
        self.optimizers=[]
        self.device = settings.device
        self.transform=torch.tensor(transform,dtype=torch.float32,device=self.device) if transform is not None else None
        self.transform_optim=_TRANSFORM_OPTIMIZER(self.transform,settings.retrieve(self.id)[0])
        self.optimizers.append(self.transform_optim)
        self.proc_appearance(appearance,settings.retrieve(self.id)[0])

    def tftostring(self):
        return self.transform_optim.tfToString()

    def appearanceToString(self):
        appstring=""
        for key,value in self.appearance.items():
            if key in ["fill", "stroke"]:
                #a paint-type value
                if value[0] == "none":
                    appstring+="{}:none;".format(key)
                elif value[0] == "solid":
                    appstring += "{}:{};".format(key,_RGB_TO_STRING(value[1]))
                elif value[0] == "url":
                    appstring += "{}:url(#{});".format(key,value[1].id)
                    #appstring += "{}:{};".format(key,"#ff00ff")
            elif key in ["opacity", "fill-opacity", "stroke-opacity", "stroke-width", "fill-rule"]:
                appstring+="{}:{};".format(key,value)
            else:
                raise ValueError("Don't know how to write appearance parameter '{}'".format(key))
        return appstring


    def write_xml_common_attrib(self,node,tfname="transform"):
        if self.transform is not None:
            node.set(tfname,self.tftostring())
        if len(self.appearance)>0:
            node.set('style',self.appearanceToString())
        if self.id is not None:
            node.set('id',self.id)


    def proc_appearance(self,appearance,optim_params):
        self.appearance=appearance
        for key, value in appearance.items():
            if key == "fill" or key == "stroke":
                if optim_params["optimize_color"] and value[0]=="solid":
                    value[1].requires_grad_(True)
                    self.optimizers.append(_COLOR_OPTIMIZER(value[1],SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["color_lr"]))
            elif key == "fill-opacity" or key == "stroke-opacity" or key == "opacity":
                if optim_params["optimize_alpha"]:
                    value[1].requires_grad_(True)
                    self.optimizers.append(
                        _COLOR_OPTIMIZER(
                            value[1],
                            SvgOptimizationSettings.optims[optim_params["optimizer"]],
                            optim_params["alpha_lr"],
                        )
                    )
            elif key == "fill-rule" or key == "stroke-width":
                pass
            else:
                raise RuntimeError("Unrecognized appearance key '{}'".format(key))

    def prop_transform(self,intform):
        return intform.matmul(self.transform_optim.get_transform()) if self.transform is not None else intform

    def prop_appearance(self,inappearance):
        outappearance=copy.copy(inappearance)
        for key,value in self.appearance.items():
            if key == "fill":
                #gets replaced
                outappearance[key]=value
            elif key == "fill-opacity":
                #gets multiplied
                outappearance[key] = outappearance[key]*value
            elif key == "fill-rule":
                #gets replaced
                outappearance[key] = value
            elif key =="opacity":
                # gets multiplied
                outappearance[key] = outappearance[key]*value
            elif key == "stroke":
                # gets replaced
                outappearance[key] = value
            elif key == "stroke-opacity":
                # gets multiplied
                outappearance[key] = outappearance[key]*value
            elif key =="stroke-width":
                # gets replaced
                outappearance[key] = value
            else:
                raise RuntimeError("Unrecognized appearance key '{}'".format(key))
        return outappearance

    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()
        for child in self.children:
            child.zero_grad()

    def step(self):
        for optim in self.optimizers:
            optim.step()
        for child in self.children:
            child.step()

    def get_type(self):
        return "Generic node"

    def is_shape(self):
        return False

    def build_scene(self,shapes,shape_groups,transform,appearance):
        raise NotImplementedError("Abstract SvgNode cannot recurse")

class GroupNode(SvgNode):
    def __init__(self, id, transform, appearance,settings):
        super().__init__(id, transform, appearance,settings)

    def get_type(self):
        return "Group node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        outtf=self.prop_transform(transform)
        outapp=self.prop_appearance(appearance)
        for child in self.children:
            child.build_scene(shapes,shape_groups,outtf,outapp)

    def write_xml(self, parent):
        elm=etree.SubElement(parent,"g")
        self.write_xml_common_attrib(elm)

        for child in self.children:
            child.write_xml(elm)

class RootNode(SvgNode):
    def __init__(self, id, transform, appearance,settings):
        super().__init__(id, transform, appearance,settings)

    def write_xml(self,document):
        elm=etree.Element('svg')
        self.write_xml_common_attrib(elm)
        elm.set("version","2.0")
        elm.set("width",str(document.canvas[0]))
        elm.set("height", str(document.canvas[1]))
        elm.set("xmlns","http://www.w3.org/2000/svg")
        elm.set("xmlns:xlink","http://www.w3.org/1999/xlink")
        #write definitions before we write any children
        document.write_defs(elm)

        #write the children
        for child in self.children:
            child.write_xml(elm)

        return elm

    def get_type(self):
        return "Root node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        outtf = self.prop_transform(transform).to(self.device)
        for child in self.children:
            child.build_scene(shapes,shape_groups,outtf,appearance)

    @staticmethod
    def get_default_appearance(device):
        default_appearance = {"fill": ("solid", torch.tensor([0., 0., 0.],device=device)),
                              "fill-opacity": torch.tensor([1.],device=device),
                              "fill-rule": "nonzero",
                              "opacity": torch.tensor([1.],device=device),
                              "stroke": ("none", None),
                              "stroke-opacity": torch.tensor([1.],device=device),
                              "stroke-width": torch.tensor([0.],device=device)}
        return default_appearance

    @staticmethod
    def get_default_transform():
        return torch.eye(3)



class ShapeNode(SvgNode):
    def __init__(self, id, transform, appearance,settings):
        super().__init__(id, transform, appearance,settings)

    def get_type(self):
        return "Generic shape node"

    def is_shape(self):
        return True

    def construct_paint(self,value,combined_opacity,transform):
        if value[0]   == "none":
            return None
        elif value[0] == "solid":
            return torch.cat([value[1],combined_opacity]).to(self.device)
        elif value[0] == "url":
            #get the gradient object from this node
            return value[1].getGrad(combined_opacity,transform)
        else:
            raise ValueError("Unknown paint value type '{}'".format(value[0]))

    def make_shape_group(self,appearance,transform,num_shapes,num_subobjects):
        fill=self.construct_paint(appearance["fill"],appearance["opacity"]*appearance["fill-opacity"],transform)
        stroke=self.construct_paint(appearance["stroke"],appearance["opacity"]*appearance["stroke-opacity"],transform)
        sg = pydiffvg.ShapeGroup(shape_ids=torch.tensor(range(num_shapes, num_shapes + num_subobjects)),
                                 fill_color=fill,
                                 use_even_odd_rule=appearance["fill-rule"]=="evenodd",
                                 stroke_color=stroke,
                                 shape_to_canvas=transform,
                                 id=self.id)
        return sg

class PathNode(ShapeNode):
    def __init__(self, id, transform, appearance,settings, paths):
        super().__init__(id, transform, appearance,settings)
        self.proc_paths(paths,settings.retrieve(self.id)[0])

    def proc_paths(self,paths,optim_params):
        self.paths=paths
        if optim_params["paths"]["optimize_points"]:
            ptlist=[]
            for path in paths:
                ptlist.append(path.points.requires_grad_(True))
            self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]](ptlist,lr=optim_params["paths"]["shape_lr"]))

    def get_type(self):
        return "Path node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        applytf=self.prop_transform(transform)
        applyapp = self.prop_appearance(appearance)
        sg=self.make_shape_group(applyapp,applytf,len(shapes),len(self.paths))
        for path in self.paths:
            disp_path=pydiffvg.Path(path.num_control_points,path.points,path.is_closed,applyapp["stroke-width"],path.id)
            shapes.append(disp_path)
        shape_groups.append(sg)

    def path_to_string(self,path):
        path_string = "M {},{} ".format(path.points[0][0].item(), path.points[0][1].item())
        idx = 1
        numpoints = path.points.shape[0]
        for type in path.num_control_points:
            toproc = type + 1
            if type == 0:
                # add line
                path_string += "L "
            elif type == 1:
                # add quadric
                path_string += "Q "
            elif type == 2:
                # add cubic
                path_string += "C "
            while toproc > 0:
                path_string += "{},{} ".format(path.points[idx % numpoints][0].item(),
                                               path.points[idx % numpoints][1].item())
                idx += 1
                toproc -= 1
        if path.is_closed:
            path_string += "Z "

        return path_string

    def paths_string(self):
        pstr=""
        for path in self.paths:
            pstr+=self.path_to_string(path)
        return pstr

    def write_xml(self, parent):
        elm = etree.SubElement(parent, "path")
        self.write_xml_common_attrib(elm)
        elm.set("d",self.paths_string())

        for child in self.children:
            child.write_xml(elm)

class RectNode(ShapeNode):
    def __init__(self, id, transform, appearance,settings, rect):
        super().__init__(id, transform, appearance,settings)
        self.rect=torch.tensor(rect,dtype=torch.float,device=settings.device)
        optim_params=settings.retrieve(self.id)[0]
        #borrowing path settings for this
        if optim_params["paths"]["optimize_points"]:
            self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.rect],lr=optim_params["paths"]["shape_lr"]))

    def get_type(self):
        return "Rect node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        applytf=self.prop_transform(transform)
        applyapp = self.prop_appearance(appearance)
        sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
        shapes.append(pydiffvg.Rect(self.rect[0:2],self.rect[0:2]+self.rect[2:4],applyapp["stroke-width"],self.id))
        shape_groups.append(sg)

    def write_xml(self, parent):
        elm = etree.SubElement(parent, "rect")
        self.write_xml_common_attrib(elm)
        elm.set("x",str(self.rect[0]))
        elm.set("y", str(self.rect[1]))
        elm.set("width", str(self.rect[2]))
        elm.set("height", str(self.rect[3]))

        for child in self.children:
            child.write_xml(elm)

class CircleNode(ShapeNode):
    def __init__(self, id, transform, appearance,settings, rect):
        super().__init__(id, transform, appearance,settings)
        self.circle=torch.tensor(rect,dtype=torch.float,device=settings.device)
        optim_params=settings.retrieve(self.id)[0]
        #borrowing path settings for this
        if optim_params["paths"]["optimize_points"]:
            self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.circle],lr=optim_params["paths"]["shape_lr"]))

    def get_type(self):
        return "Circle node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        applytf=self.prop_transform(transform)
        applyapp = self.prop_appearance(appearance)
        sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
        shapes.append(pydiffvg.Circle(self.circle[2],self.circle[0:2],applyapp["stroke-width"],self.id))
        shape_groups.append(sg)

    def write_xml(self, parent):
        elm = etree.SubElement(parent, "circle")
        self.write_xml_common_attrib(elm)
        elm.set("cx",str(self.circle[0]))
        elm.set("cy", str(self.circle[1]))
        elm.set("r", str(self.circle[2]))

        for child in self.children:
            child.write_xml(elm)


class EllipseNode(ShapeNode):
    def __init__(self, id, transform, appearance,settings, ellipse):
        super().__init__(id, transform, appearance,settings)
        self.ellipse=torch.tensor(ellipse,dtype=torch.float,device=settings.device)
        optim_params=settings.retrieve(self.id)[0]
        #borrowing path settings for this
        if optim_params["paths"]["optimize_points"]:
            self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.ellipse],lr=optim_params["paths"]["shape_lr"]))

    def get_type(self):
        return "Ellipse node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        applytf=self.prop_transform(transform)
        applyapp = self.prop_appearance(appearance)
        sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
        shapes.append(pydiffvg.Ellipse(self.ellipse[2:4],self.ellipse[0:2],applyapp["stroke-width"],self.id))
        shape_groups.append(sg)

    def write_xml(self, parent):
        elm = etree.SubElement(parent, "ellipse")
        self.write_xml_common_attrib(elm)
        elm.set("cx", str(self.ellipse[0]))
        elm.set("cy", str(self.ellipse[1]))
        elm.set("rx", str(self.ellipse[2]))
        elm.set("ry", str(self.ellipse[3]))

        for child in self.children:
            child.write_xml(elm)

class PolygonNode(ShapeNode):
    def __init__(self, id, transform, appearance,settings, points):
        super().__init__(id, transform, appearance,settings)
        self.points=points
        optim_params=settings.retrieve(self.id)[0]
        #borrowing path settings for this
        if optim_params["paths"]["optimize_points"]:
            self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.points],lr=optim_params["paths"]["shape_lr"]))

    def get_type(self):
        return "Polygon node"

    def build_scene(self,shapes,shape_groups,transform,appearance):
        applytf=self.prop_transform(transform)
        applyapp = self.prop_appearance(appearance)
        sg=self.make_shape_group(applyapp,applytf,len(shapes),1)
        shapes.append(pydiffvg.Polygon(self.points,True,applyapp["stroke-width"],self.id))
        shape_groups.append(sg)

    def point_string(self):
        ret=""
        for i in range(self.points.shape[0]):
            pt=self.points[i,:]
            #assert pt.shape == (1,2)
            ret+= str(pt[0])+","+str(pt[1])+" "
        return ret

    def write_xml(self, parent):
        elm = etree.SubElement(parent, "polygon")
        self.write_xml_common_attrib(elm)
        elm.set("points",self.point_string())

        for child in self.children:
            child.write_xml(elm)

class GradientNode(SvgNode):
    def __init__(self, id, transform,settings,begin,end,offsets,stops,href):
        super().__init__(id, transform, {},settings)
        self.optim=_GRADIENT_OPTIMIZER(begin, end, offsets, stops, settings.retrieve(id)[0])
        self.optimizers.append(self.optim)
        self.href=href

    def is_ref(self):
        return self.href is not None

    def get_type(self):
        return "Gradient node"

    def get_stops(self):
        _, _, offsets, stops=self.optim.get_vals()
        return offsets, stops

    def get_points(self):
        begin, end, _, _ =self.optim.get_vals()
        return begin, end

    def write_xml(self, parent):
        elm = etree.SubElement(parent, "linearGradient")
        self.write_xml_common_attrib(elm,tfname="gradientTransform")

        begin, end, offsets, stops = self.optim.get_vals()

        if self.href is None:
            #we have stops
            for idx, offset in enumerate(offsets):
                stop=etree.SubElement(elm,"stop")
                stop.set("offset",str(offset.item()))
                stop.set("stop-color",_RGB_TO_STRING(stops[idx,0:3]))
                stop.set("stop-opacity",str(stops[idx,3].item()))
        else:
            elm.set('xlink:href', "#{}".format(self.href.id))

        if begin is not None and end is not None:
            #no stops
            elm.set('x1', str(begin[0].item()))
            elm.set('y1', str(begin[1].item()))
            elm.set('x2', str(end[0].item()))
            elm.set('y2', str(end[1].item()))

            # magic value to make this work
            elm.set("gradientUnits", "userSpaceOnUse")

        for child in self.children:
            child.write_xml(elm)

    def getGrad(self,combined_opacity,transform):
        if self.is_ref():
            offsets, stops=self.href.get_stops()
        else:
            offsets, stops=self.get_stops()

        stops=stops.clone()
        stops[:,3]*=combined_opacity

        begin,end = self.get_points()

        applytf=self.prop_transform(transform)
        begin=_TRANSFORM_TOOLS.transformPoints(begin.unsqueeze(0),applytf).squeeze()
        end = _TRANSFORM_TOOLS.transformPoints(end.unsqueeze(0), applytf).squeeze()

        return pydiffvg.LinearGradient(begin, end, offsets, stops)

__all__ = [
    'SvgNode',
    'GroupNode',
    'RootNode',
    'ShapeNode',
    'PathNode',
    'RectNode',
    'CircleNode',
    'EllipseNode',
    'PolygonNode',
    'GradientNode',
    'configure_scene_graph',
]
