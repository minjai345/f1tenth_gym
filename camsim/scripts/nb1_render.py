"""NB1: 트랙 + 렌더러 + GT + IPM 왕복. gym 불필요."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.getcwd())
from camsim import config, camera, track, render, gt

cfg = config.load()
trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
track.save(trk, "track.npz")
H_g2i, H_i2g = camera.build(cfg)
print("focal px", camera.focal_px(cfg), "track length m", trk.length, "quads", len(trk.quads))

os.makedirs("out", exist_ok=True)
rng = np.random.default_rng(0)
for n in range(6):
    pose = gt.sample_pose(trk, cfg, rng)
    img = render.render(pose, trk.quads, None, H_g2i, cfg)
    render.draw_points(img, gt.waypoints_ahead(pose, trk, cfg), H_g2i)
    cv2.imwrite(f"out/nb1_sample_{n}.png", img)

# IPM 왕복: 렌더 -> 테이프 픽셀 -> H_i2g -> 지면. 원래 테이프와 얼마나 맞는가
pose = np.array([*trk.center[50], trk.heading[50]])
img = render.render(pose, trk.quads, None, H_g2i, cfg)
vs, us = np.where(np.all(img == cfg.lane.color_tape, axis=-1))
g = camera.project(H_i2g, np.column_stack([us, vs]).astype(float))
qv = render.to_vehicle(pose, trk.quads).reshape(-1, 2)
d = np.sqrt(((g[:, None, :] - qv[None, :, :]) ** 2).sum(-1)).min(1)
print(f"IPM round trip: median {np.median(d)*100:.1f} cm, 95% {np.percentile(d,95)*100:.1f} cm")

# BEV 그림 (0.2~4 m x -1.5~1.5 m, 5 mm/px)
bev = np.full((600, 760, 3), 255, np.uint8)
px = ((g[:, 0] - 0.2) / 0.005).astype(int); py = ((1.5 - g[:, 1]) / 0.005).astype(int)
ok = (px >= 0) & (px < 760) & (py >= 0) & (py < 600)
bev[py[ok], px[ok]] = (0, 0, 255)
cv2.imwrite("out/nb1_bev.png", cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE))
print("wrote out/nb1_*.png")
