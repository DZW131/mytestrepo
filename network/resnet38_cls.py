import torch
import torch.nn as nn
import torch.nn.functional as F
import network.resnet38d
from network.hst.context import (
    apply_contextual_homogenization,
    build_context_conv,
)
from network.hst.hst_rectifier import HSTConfig, HSTRectifier


class HFRM(nn.Module):
    def __init__(self, in_channels, deep_channels=4096, context_kernel=15):
        super(HFRM, self).__init__()
        

        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, deep_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(deep_channels // 8, in_channels, bias=False),
            nn.Sigmoid()
        )
        

        self.context_conv = build_context_conv(in_channels, context_kernel)
        

        self.gamma_veto = nn.Parameter(torch.zeros(1))
        self.gamma_context = nn.Parameter(torch.zeros(1))

    def forward(self, feat_nong, feat_deep):
        B, C, H, W = feat_nong.size()
        

        global_dna = F.adaptive_avg_pool2d(feat_deep, 1).view(B, -1) # [B, 4096]
        veto_weights = self.veto_mlp(global_dna).view(B, C, 1, 1)    
        

        feat_vetoed = feat_nong * veto_weights
        
        # 2. Contextual Homogenization Processing

        feat_smoothed = apply_contextual_homogenization(
            self.context_conv, feat_nong
        )
        
        # 3. (Residual sum)
        feat_rectified = feat_nong + \
                         self.gamma_veto * feat_vetoed + \
                         self.gamma_context * feat_smoothed
                         
        return feat_rectified

# =========================================================================
# 2.  (Main Training Network)
# =========================================================================
class Net(network.resnet38d.Net):
    def __init__(self, n_class, rectifier_type="hfrm", hst_config=None):
        super().__init__()

        self.rectifier_type = rectifier_type.lower()
        if self.rectifier_type not in {"hfrm", "hst"}:
            raise ValueError(
                "rectifier_type must be either 'hfrm' or 'hst', "
                f"got {rectifier_type!r}"
            )

        self.dropout7 = torch.nn.Dropout2d(0.5)

        if self.rectifier_type == "hfrm":
            self.hfrm_56 = HFRM(in_channels=256, deep_channels=4096, context_kernel=15)
            self.hfrm_28_1 = HFRM(in_channels=512, deep_channels=4096, context_kernel=15)
            self.hfrm_28_2 = HFRM(in_channels=1024, deep_channels=4096, context_kernel=15)
            rectifier_layers = [self.hfrm_56, self.hfrm_28_1, self.hfrm_28_2]
        else:
            self.hst_rectifier = HSTRectifier(HSTConfig.from_value(hst_config))
            rectifier_layers = [self.hst_rectifier]


        self.ic_56 = nn.Conv2d(256, n_class, 1)
        torch.nn.init.xavier_uniform_(self.ic_56.weight)

        self.ic1 = nn.Conv2d(512, n_class, 1)
        torch.nn.init.xavier_uniform_(self.ic1.weight)
        
        self.ic2 = nn.Conv2d(1024, n_class, 1)
        torch.nn.init.xavier_uniform_(self.ic2.weight)

        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        

        self.from_scratch_layers = [
            self.ic_56,
            self.ic1,
            self.ic2,
            self.fc8,
            *rectifier_layers,
        ]
        self.pool = nn.MaxPool2d(2, 2)

    def _extract_backbone_features(self, x):
        x = self.conv1a(x)
        x = self.b2(x); x = self.b2_1(x); x = self.b2_2(x)
        
        x = self.b3(x); x = self.b3_1(x); x = self.b3_2(x)
        feat_56 = x  

        x = self.b4(x); x = self.b4_1(x); x = self.b4_2(x); x = self.b4_3(x); x = self.b4_4(x); x = self.b4_5(x)
        feat_28_1 = F.relu(self.bn45(x)) 
        
        x, _ = self.b5(x, get_x_bn_relu=True); x = self.b5_1(x); x = self.b5_2(x)
        feat_28_2 = F.relu(self.bn52(x)) 
        
        x, _ = self.b6(x, get_x_bn_relu=True); x = self.b7(x)
        feat_deep = F.relu(self.bn7(x)) 

        return feat_56, feat_28_1, feat_28_2, feat_deep

    def _rectify_features(
        self, feat_56, feat_28_1, feat_28_2, feat_deep
    ):
        if self.rectifier_type == "hfrm":
            feat_56_rectified = self.hfrm_56(feat_56, feat_deep)
            feat_28_1_rectified = self.hfrm_28_1(feat_28_1, feat_deep)
            feat_28_2_rectified = self.hfrm_28_2(feat_28_2, feat_deep)
            diagnostics = {
                "base_features": {
                    "stage1": feat_56,
                    "stage2": feat_28_1,
                    "stage3": feat_28_2,
                    "deep": feat_deep,
                },
                "rectified_features": {
                    "stage1": feat_56_rectified,
                    "stage2": feat_28_1_rectified,
                    "stage3": feat_28_2_rectified,
                    "deep": feat_deep,
                },
                "correction_states": {},
                "semantic_gates": {},
                "context_features": {},
            }
        else:
            diagnostics = self.hst_rectifier(
                feat_56, feat_28_1, feat_28_2, feat_deep
            )
            rectified = diagnostics["rectified_features"]
            feat_56_rectified = rectified["stage1"]
            feat_28_1_rectified = rectified["stage2"]
            feat_28_2_rectified = rectified["stage3"]

        return (
            feat_56_rectified,
            feat_28_1_rectified,
            feat_28_2_rectified,
            diagnostics,
        )

    def _forward_impl(self, x, return_diagnostics=False):
        feat_56, feat_28_1, feat_28_2, feat_deep = \
            self._extract_backbone_features(x)
        (
            feat_56_rectified,
            feat_28_1_rectified,
            feat_28_2_rectified,
            diagnostics,
        ) = self._rectify_features(
            feat_56, feat_28_1, feat_28_2, feat_deep
        )


        cam_56 = self.ic_56(feat_56_rectified)
        cam_28_1 = self.ic1(feat_28_1_rectified)
        cam_28_2 = self.ic2(feat_28_2_rectified)
        
        feat_deep_drop = self.dropout7(feat_deep)
        cam_deep = self.fc8(feat_deep_drop)

        batch_size = feat_deep.size(0)
        out_56 = F.avg_pool2d(cam_56, kernel_size=(cam_56.size(2), cam_56.size(3)), padding=0).view(batch_size, -1)
        out_28_1 = F.avg_pool2d(cam_28_1, kernel_size=(cam_28_1.size(2), cam_28_1.size(3)), padding=0).view(batch_size, -1)
        out_28_2 = F.avg_pool2d(cam_28_2, kernel_size=(cam_28_2.size(2), cam_28_2.size(3)), padding=0).view(batch_size, -1)
        out_deep = F.avg_pool2d(cam_deep, kernel_size=(cam_deep.size(2), cam_deep.size(3)), padding=0).view(batch_size, -1)

        y_deep = torch.sigmoid(out_deep)

        outputs = (
            out_56,
            out_28_1,
            out_28_2,
            out_deep,
            y_deep,
            cam_56,
            cam_28_1,
            cam_28_2,
            cam_deep,
            feat_56_rectified,
        )
        if return_diagnostics:
            diagnostics["cam_logits"] = {
                "stage1": cam_56,
                "stage2": cam_28_1,
                "stage3": cam_28_2,
                "deep": cam_deep,
            }
            return outputs, diagnostics
        return outputs

    def forward(self, x):
        return self._forward_impl(x, return_diagnostics=False)

    def forward_with_diagnostics(self, x):
        """Return the unchanged public outputs plus analysis-only tensors."""
        return self._forward_impl(x, return_diagnostics=True)

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        def is_scratch(m): 
            for layer in self.from_scratch_layers:
                if layer is m or m in layer.modules():
                    return True
            return False

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    groups[2].append(m.weight) if is_scratch(m) else groups[0].append(m.weight)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    groups[3].append(m.bias) if is_scratch(m) else groups[1].append(m.bias)
            elif 'Norm' in m.__class__.__name__ or 'BatchNorm2d' in m.__class__.__name__:
                for name, param in m.named_parameters(recurse=False):
                    if param.requires_grad:
                        if is_scratch(m):
                            if 'bias' in name or 'beta' in name: groups[3].append(param)
                            else: groups[2].append(param)
                        else:
                            if 'bias' in name or 'beta' in name: groups[1].append(param)
                            else: groups[0].append(param)

            elif isinstance(m, HFRM):
                groups[2].append(m.gamma_veto)
                groups[2].append(m.gamma_context)
            elif isinstance(m, HSTRectifier):
                groups[2].extend(m.residual_scale_parameters())
                
        return groups

    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, nn.Conv2d) and m.groups != m.in_channels: 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Net_CAM(Net):
    def __init__(self, n_class, rectifier_type="hfrm", hst_config=None):
        super().__init__(
            n_class,
            rectifier_type=rectifier_type,
            hst_config=hst_config,
        )

    def forward(self, x):
        out_56, out_28_1, out_28_2, out_deep, y_deep, _, _, _, _, _ = super().forward(x)
        return y_deep

    def forward_cam(self, x):
        feat_56, feat_28_1, feat_28_2, feat_deep = \
            self._extract_backbone_features(x)
        (
            feat_56_rectified,
            feat_28_1_rectified,
            feat_28_2_rectified,
            _,
        ) = self._rectify_features(
            feat_56, feat_28_1, feat_28_2, feat_deep
        )


        cam_56 = F.relu(self.ic_56(feat_56_rectified))
        cam_28_1 = F.relu(self.ic1(feat_28_1_rectified))
        cam_28_2 = F.relu(self.ic2(feat_28_2_rectified))
        cam_deep = F.relu(self.fc8(feat_deep)) 
        
        out_deep = F.avg_pool2d(self.fc8(feat_deep), kernel_size=(feat_deep.size(2), feat_deep.size(3)), padding=0).view(feat_deep.size(0), -1)
        y_deep = torch.sigmoid(out_deep)

        return cam_56, cam_28_1, cam_28_2, cam_deep, y_deep
