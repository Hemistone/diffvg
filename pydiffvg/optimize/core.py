import copy
import math
import numpy as np
import torch
import pydiffvg
import xml.etree.ElementTree as etree

from .settings import SvgOptimizationSettings
from . import scene_graph as _scene_graph
from .transforms import TransformTools as _TransformTools
from .parser import SvgParserMixin
from .writer import SvgWriterMixin


class OptimizableSvg(SvgParserMixin, SvgWriterMixin):
    TransformTools = _TransformTools

    #region suboptimizers

    #optimizes color, but really any tensor that needs to stay between 0 and 1 per-entry
    class ColorOptimizer:
        def __init__(self,tensor,optim_type,lr):
            self.tensor=tensor
            self.optim=optim_type([tensor],lr=lr)

        def zero_grad(self):
            self.optim.zero_grad()

        def step(self):
            self.optim.step()
            self.tensor.data.clamp_(min=1e-4,max=1.)

    #optimizes gradient stop positions
    class StopOptimizer:
        def __init__(self,stops,optim_type,lr):
            self.stops=stops
            self.optim=optim_type([stops],lr=lr)

        def zero_grad(self):
            self.optim.zero_grad()

        def step(self):
            self.optim.step()
            self.stops.data.clamp_(min=0., max=1.)
            self.stops.data, _ = self.stops.sort()
            self.stops.data[0] = 0.
            self.stops.data[-1]=1.

    #optimizes gradient: stop, positions, colors+opacities, locations
    class GradientOptimizer:
        def __init__(self, begin, end, offsets, stops, optim_params):
            self.begin=begin.clone().detach() if begin is not None else None
            self.end=end.clone().detach() if end is not None else None
            self.offsets=offsets.clone().detach() if offsets is not None else None
            self.stop_colors=stops[:,0:3].clone().detach() if stops is not None else None
            self.stop_alphas=stops[:,3].clone().detach() if stops is not None else None
            self.optimizers=[]

            if optim_params["gradients"]["optimize_stops"] and self.offsets is not None:
                self.offsets.requires_grad_(True)
                self.optimizers.append(OptimizableSvg.StopOptimizer(self.offsets,SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["gradients"]["stop_lr"]))
            if optim_params["gradients"]["optimize_color"] and self.stop_colors is not None:
                self.stop_colors.requires_grad_(True)
                self.optimizers.append(OptimizableSvg.ColorOptimizer(self.stop_colors,SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["gradients"]["color_lr"]))
            if optim_params["gradients"]["optimize_alpha"] and self.stop_alphas is not None:
                self.stop_alphas.requires_grad_(True)
                self.optimizers.append(OptimizableSvg.ColorOptimizer(self.stop_alphas,SvgOptimizationSettings.optims[optim_params["optimizer"]],optim_params["gradients"]["alpha_lr"]))
            if optim_params["gradients"]["optimize_location"] and self.begin is not None and self.end is not None:
                self.begin.requires_grad_(True)
                self.end.requires_grad_(True)
                self.optimizers.append(SvgOptimizationSettings.optims[optim_params["optimizer"]]([self.begin,self.end],lr=optim_params["gradients"]["location_lr"]))


        def get_vals(self):
            return self.begin, self.end, self.offsets, torch.cat((self.stop_colors,self.stop_alphas.unsqueeze(1)),1) if self.stop_colors is not None and self.stop_alphas is not None else None

        def zero_grad(self):
            for optim in self.optimizers:
                optim.zero_grad()

        def step(self):
            for optim in self.optimizers:
                optim.step()

    class TransformOptimizer:
        def __init__(self,transform,optim_params):
            self.transform=transform
            self.optimizes=optim_params["transforms"]["optimize_transforms"] and transform is not None
            self.params=copy.deepcopy(optim_params)
            self.transform_mode=optim_params["transforms"]["transform_mode"]

            if self.optimizes:
                optimvars=[]
                self.residual=None
                lr=optim_params["transforms"]["transform_lr"]
                tmult=optim_params["transforms"]["translation_mult"]
                decomp,props=OptimizableSvg.TransformTools.check_and_decomp(transform.cpu().numpy())
                if self.transform_mode=="move":
                    #only translation and rotation should be set
                    if props.has_scale or props.has_shear or props.has_mirror:
                        print("Warning: set to optimize move only, but input transform has residual scale or shear")
                        self.residual=self.transform.clone().detach().requires_grad_(False)
                        self.Theta=torch.tensor(0,dtype=torch.float32,requires_grad=True,device=transform.device)
                        self.translation=torch.tensor([0, 0],dtype=torch.float32,requires_grad=True,device=transform.device)
                    else:
                        self.residual=None
                        self.Theta=torch.tensor(decomp.theta,dtype=torch.float32,requires_grad=True,device=transform.device)
                        self.translation=torch.tensor(decomp.translate,dtype=torch.float32,requires_grad=True,device=transform.device)
                    optimvars+=[{'params':x,'lr':lr} for x in [self.Theta]]+[{'params':self.translation,'lr':lr*tmult}]
                elif self.transform_mode=="rigid":
                    #only translation, rotation, and uniform scale should be set
                    if props.has_shear or props.has_mirror or not props.scale_uniform:
                        print("Warning: set to optimize rigid transform only, but input transform has residual shear, mirror or non-uniform scale")
                        self.residual = self.transform.clone().detach().requires_grad_(False)
                        self.Theta = torch.tensor(0, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale=torch.tensor(1, dtype=torch.float32, requires_grad=True,device=transform.device)
                    else:
                        self.residual = None
                        self.Theta = torch.tensor(decomp.theta, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor(decomp.translate, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale = torch.tensor(decomp.scale[0], dtype=torch.float32, requires_grad=True,device=transform.device)
                    optimvars += [{'params':x,'lr':lr} for x in [self.Theta, self.scale]]+[{'params':self.translation,'lr':lr*tmult}]
                elif self.transform_mode=="similarity":
                    if props.has_shear or not props.scale_uniform:
                        print("Warning: set to optimize rigid transform only, but input transform has residual shear or non-uniform scale")
                        self.residual = self.transform.clone().detach().requires_grad_(False)
                        self.Theta = torch.tensor(0, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale=torch.tensor(1, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale_sign=torch.tensor(1,dtype=torch.float32,requires_grad=False,device=transform.device)
                    else:
                        self.residual = None
                        self.Theta = torch.tensor(decomp.theta, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.translation = torch.tensor(decomp.translate, dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale = torch.tensor(decomp.scale[0], dtype=torch.float32, requires_grad=True,device=transform.device)
                        self.scale_sign = torch.tensor(np.sign(decomp.scale[0]*decomp.scale[1]), dtype=torch.float32, requires_grad=False,device=transform.device)
                    optimvars += [{'params':x,'lr':lr} for x in [self.Theta, self.scale]]+[{'params':self.translation,'lr':lr*tmult}]
                elif self.transform_mode=="affine":
                    self.Theta = torch.tensor(decomp.theta, dtype=torch.float32, requires_grad=True,device=transform.device)
                    self.translation = torch.tensor(decomp.translate, dtype=torch.float32, requires_grad=True,device=transform.device)
                    self.scale = torch.tensor(decomp.scale, dtype=torch.float32, requires_grad=True,device=transform.device)
                    self.shear = torch.tensor(decomp.shear, dtype=torch.float32, requires_grad=True,device=transform.device)
                    optimvars += [{'params':x,'lr':lr} for x in [self.Theta, self.scale, self.shear]]+[{'params':self.translation,'lr':lr*tmult}]
                else:
                    raise ValueError("Unrecognized transform mode '{}'".format(self.transform_mode))
                self.optimizer=SvgOptimizationSettings.optims[optim_params["optimizer"]](optimvars)

        def get_transform(self):
            if not self.optimizes:
                return self.transform
            else:
                if self.transform_mode == "move":
                    composed=OptimizableSvg.TransformTools.recompose(self.Theta,torch.tensor([1.],device=self.Theta.device),torch.tensor(0.,device=self.Theta.device),self.translation)
                    return self.residual.mm(composed) if self.residual is not None else composed
                elif self.transform_mode == "rigid":
                    composed = OptimizableSvg.TransformTools.recompose(self.Theta, self.scale, torch.tensor(0.,device=self.Theta.device),
                                                                       self.translation)
                    return self.residual.mm(composed) if self.residual is not None else composed
                elif self.transform_mode == "similarity":
                    composed=OptimizableSvg.TransformTools.recompose(self.Theta, torch.cat((self.scale,self.scale*self.scale_sign)),torch.tensor(0.,device=self.Theta.device),self.translation)
                    return self.residual.mm(composed) if self.residual is not None else composed
                elif self.transform_mode == "affine":
                    composed = OptimizableSvg.TransformTools.recompose(self.Theta, self.scale, self.shear, self.translation)
                    return composed
                else:
                    raise ValueError("Unrecognized transform mode '{}'".format(self.transform_mode))

        def tfToString(self):
            if self.transform is None:
                return None
            elif not self.optimizes:
                return OptimizableSvg.TransformTools.tf_to_string(self.transform)
            else:
                if self.transform_mode == "move":
                    str=OptimizableSvg.TransformTools.decomp_to_string((self.Theta,torch.tensor([1.]),torch.tensor(0.),self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "rigid":
                    str = OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self.scale, torch.tensor(0.),
                                                                       self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "similarity":
                    str=OptimizableSvg.TransformTools.decomp_to_string((self.Theta, torch.cat((self.scale,self.scale*self.scale_sign)),torch.tensor(0.),self.translation))
                    return (OptimizableSvg.TransformTools.tf_to_string(self.residual) if self.residual is not None else "")+" "+str
                elif self.transform_mode == "affine":
                    str = OptimizableSvg.TransformTools.decomp_to_string((self.Theta, self.scale, self.shear, self.translation))
                    return str

        def zero_grad(self):
            if self.optimizes:
                self.optimizer.zero_grad()

        def step(self):
            if self.optimizes:
                self.optimizer.step()

    #endregion

    #region Nodes
    # Scene graph node classes are defined in pydiffvg/optimize/scene_graph.py.
    # They are injected onto OptimizableSvg after configuration.
    #endregion

    def __init__(self, filename, settings=SvgOptimizationSettings(),optimize_background=False, verbose=False, device=torch.device("cpu")):
        self.settings=settings
        self.verbose=verbose
        self.device=device
        self.settings.device=device

        tree = etree.parse(filename)
        root = tree.getroot()

        #in case we need global optimization
        self.optimizers=[]
        self.background=torch.tensor([1.,1.,1.],dtype=torch.float32,requires_grad=optimize_background,device=self.device)

        if optimize_background:
            p=settings.retrieve("default")[0]
            self.optimizers.append(OptimizableSvg.ColorOptimizer(self.background,SvgOptimizationSettings.optims[p["optimizer"]],p["color_lr"]))

        self.defs={}

        self.depth=0

        self.dirty=True
        self.scene=None

        self.parseRoot(root)

    recognised_shapes=["path","circle","rect","ellipse","polygon"]

    #region core functionality
    def build_scene(self):
        if self.dirty:
            shape_groups=[]
            shapes=[]
            self.root.build_scene(shapes,shape_groups,OptimizableSvg.RootNode.get_default_transform().to(self.device),OptimizableSvg.RootNode.get_default_appearance(self.device))
            self.scene=(self.canvas[0],self.canvas[1],shapes,shape_groups)
            self.dirty=False
        return self.scene

    def zero_grad(self):
        self.root.zero_grad()
        for optim in self.optimizers:
            optim.zero_grad()
        for item in self.defs.values():
            if issubclass(item.__class__,OptimizableSvg.SvgNode):
                item.zero_grad()

    def render(self,scale=None,seed=0):
        #render at native resolution
        scene = self.build_scene()
        scene_args = pydiffvg.RenderFunction.serialize_scene(*scene)
        render = pydiffvg.RenderFunction.apply
        out_size=(scene[0],scene[1]) if scale is None else (int(scene[0]*scale),int(scene[1]*scale))
        img = render(out_size[0],  # width
                     out_size[1],  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     seed,  # seed
                     None, # background_image
                     *scene_args)
        return img

    def step(self):
        self.dirty=True
        self.root.step()
        for optim in self.optimizers:
            optim.step()
        for item in self.defs.values():
            if issubclass(item.__class__, OptimizableSvg.SvgNode):
                item.step()
    #endregion

    #region reporting

    def offset_str(self,s):
        return ("\t"*self.depth)+s

    def reportSkippedAttribs(self, node, non_skipped=[]):
        skipped=set([k for k in node.attrib.keys() if not OptimizableSvg.is_namespace(k)])-set(non_skipped)
        if len(skipped)>0:
            tag=OptimizableSvg.remove_namespace(node.tag) if "id" not in node.attrib else "{}#{}".format(OptimizableSvg.remove_namespace(node.tag),node.attrib["id"])
            print(self.offset_str("Warning: Skipping the following attributes of node '{}': {}".format(tag,", ".join(["'{}'".format(atr) for atr in skipped]))))

    def reportSkippedChildren(self,node,skipped):
        skipped_names=["{}#{}".format(elm.tag,elm.attrib["id"]) if "id" in elm.attrib else elm.tag for elm in skipped]
        if len(skipped)>0:
            tag = OptimizableSvg.remove_namespace(node.tag) if "id" not in node.attrib else "{}#{}".format(OptimizableSvg.remove_namespace(node.tag),
                                                                                            node.attrib["id"])
            print(self.offset_str("Warning: Skipping the following children of node '{}': {}".format(tag,", ".join(["'{}'".format(name) for name in skipped_names]))))

    #endregion

    #region parsing
    # Parsing logic supplied by SvgParserMixin
    #endregion

    #region writing
    # Writing logic supplied by SvgWriterMixin
    #endregion
_scene_graph.configure_scene_graph(
    transform_tools=OptimizableSvg.TransformTools,
    transform_optimizer=OptimizableSvg.TransformOptimizer,
    color_optimizer=OptimizableSvg.ColorOptimizer,
    gradient_optimizer=OptimizableSvg.GradientOptimizer,
    rgb_to_string=OptimizableSvg.rgb_to_string,
)

OptimizableSvg.SvgNode = _scene_graph.SvgNode
OptimizableSvg.GroupNode = _scene_graph.GroupNode
OptimizableSvg.RootNode = _scene_graph.RootNode
OptimizableSvg.ShapeNode = _scene_graph.ShapeNode
OptimizableSvg.PathNode = _scene_graph.PathNode
OptimizableSvg.RectNode = _scene_graph.RectNode
OptimizableSvg.CircleNode = _scene_graph.CircleNode
OptimizableSvg.EllipseNode = _scene_graph.EllipseNode
OptimizableSvg.PolygonNode = _scene_graph.PolygonNode
OptimizableSvg.GradientNode = _scene_graph.GradientNode

__all__ = ["SvgOptimizationSettings", "OptimizableSvg"]
