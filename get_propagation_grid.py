def get_propagation_grid(
        self, batch: int, height: int, width: int, offset: torch.Tensor, device: torch.device, img: torch.Tensor = None
):
    """Compute the offset for adaptive propagation
Args:
    batch: batch size
    height: grid height
    width: grid width
    offset: grid offset #[B,2*N_neighbors,H*W]
    device: device on which to place tensor
    img: reference images, (B, C, image_H, image_W)

Returns:
    generated grid: in the shape of [batch, propagate_neighbors*H, W, 2]
"""


# 原始的规则的矩形mask相对中心点的偏移量比如四邻域：左[-1,0] 右[1,0] 上[0,-1] 下[0,1]
# 仅支持4，8，16邻域 其它比如32可以自己实现
if self.propagate_neighbors == 4:  # if 4 neighbors to be sampled in propagation
    original_offset = [[-self.dilation, 0], [0, -self.dilation], [0, self.dilation], [self.dilation, 0]]
elif self.propagate_neighbors == 8:  # if 8 neighbors to be sampled in propagation
    original_offset = [
        [-self.dilation, -self.dilation],
        [-self.dilation, 0],
        [-self.dilation, self.dilation],
        [0, -self.dilation],
        [0, self.dilation],
        [self.dilation, -self.dilation],
        [self.dilation, 0],
        [self.dilation, self.dilation],
    ]
elif self.propagate_neighbors == 16:  # if 16 neighbors to be sampled in propagation
    original_offset = [
        [-self.dilation, -self.dilation],
        [-self.dilation, 0],
        [-self.dilation, self.dilation],
        [0, -self.dilation],
        [0, self.dilation],
        [self.dilation, -self.dilation],
        [self.dilation, 0],
        [self.dilation, self.dilation],
    ]
    for i in range(len(original_offset)):
        offset_x, offset_y = original_offset[i]
        original_offset.append([2 * offset_x, 2 * offset_y])
else:
    raise NotImplementedError

with torch.no_grad():
    y_grid, x_grid = torch.meshgrid(
        [
            torch.arange(0, height, dtype=torch.float32, device=device),
            torch.arange(0, width, dtype=torch.float32, device=device),
        ]
    )
    y_grid, x_grid = y_grid.contiguous(), x_grid.contiguous()
    y_grid, x_grid = y_grid.view(height * width), x_grid.view(height * width)
    xy = torch.stack((x_grid, y_grid))  # [2, H*W]
    xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W] 存放的是xy的像素坐标

xy_list = []
# 将前面学习得到的自适应偏移量加到原始mask的偏移量上并加上中心点坐标得到用来传播的邻域像素的像素坐标
for i in range(len(original_offset)):
    original_offset_y, original_offset_x = original_offset[i]

    offset_x_tensor = original_offset_x + offset[:, 2 * i, :].unsqueeze(1)
    offset_y_tensor = original_offset_y + offset[:, 2 * i + 1, :].unsqueeze(1)

    xy_list.append((xy + torch.cat((offset_x_tensor, offset_y_tensor), dim=1)).unsqueeze(2))

xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

del xy_list, x_grid, y_grid
# 归一化到[-1,1]
x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
del xy
grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
del x_normalized, y_normalized
grid = grid.view(batch, self.propagate_neighbors * height, width, 2)
return grid