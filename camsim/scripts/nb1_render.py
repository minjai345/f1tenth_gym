"""NB1: 트랙 + 렌더러 + GT + IPM 왕복. gym 불필요."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.getcwd())
from camsim import config, camera, track, render, gt, viz

cfg = config.load()
trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
track.save(trk, "track.npz")
H_g2i, H_i2g = camera.build(cfg)
print("focal px", camera.focal_px(cfg), "track length m", trk.length, "quads", len(trk.quads))

os.makedirs("out", exist_ok=True)
rng = np.random.default_rng(0)
mapimg = viz.MapImage(cfg.closed_loop.map_yaml)
poses = []
for n in range(6):
    pose = gt.sample_pose(trk, cfg, rng)
    wp = gt.waypoints_ahead(pose, trk, cfg)
    poses.append(pose)
    img = render.render(pose, trk.quads, None, H_g2i, cfg)
    render.draw_points(img, wp, H_g2i)
    cv2.imwrite(f"out/nb1_sample_{n}.png", img)
    # 왼쪽: 카메라 뷰(초록 = GT waypoint) | 오른쪽: 차 주변 top-down (벽, 테이프, 차, 카메라 화각, BEV 범위, GT waypoint)
    local = viz.local_view(pose, trk, wp, cfg, mapimg)
    cv2.imwrite(f"out/nb1_where_{n}.png", viz.side_by_side(img, local))

# 전체 맵: 테이프 + 샘플 6개의 위치와 heading(번호 = 샘플 번호)
overview, off = viz.draw_track_on_map(mapimg, trk, cfg)
viz.mark_poses_on_map(overview, off, mapimg, poses)
cv2.imwrite("out/nb1_map.png", overview)

# IPM 왕복: 렌더 -> 테이프 픽셀 -> H_i2g -> 지면. 원래 테이프와 얼마나 맞는가
pose = np.array([*trk.center[50], trk.heading[50]])
img = render.render(pose, trk.quads, None, H_g2i, cfg)
vs, us = np.where(np.all(img == cfg.lane.color_tape, axis=-1))
g = camera.project(H_i2g, np.column_stack([us, vs]).astype(float))
qv = render.to_vehicle(pose, trk.quads).reshape(-1, 2)
d = np.sqrt(((g[:, None, :] - qv[None, :, :]) ** 2).sum(-1)).min(1)
print(f"IPM round trip: median {np.median(d)*100:.1f} cm, 95% {np.percentile(d,95)*100:.1f} cm")

# BEV 두 장: (왼쪽) 지오메트리에서 직접 그린 정답 BEV, (오른쪽) 원근 영상을 IPM(H_i2g)으로 펴서 만든 BEV.
# 왼쪽은 카메라와 무관한 참값이고, 오른쪽이 실차 2주차 파이프라인이 실제로 보게 되는 것이다.
# 멀어질수록 오른쪽이 늘어지고 끊기는 것이 IPM 해상도 한계(문서 4.4절)다.
bev_true = render.render_bev(pose, trk.quads, cfg)
bev_ipm = render.ipm_bev(img, H_i2g, cfg)
gap = np.full((bev_true.shape[0], 20, 3), 255, np.uint8)
cv2.imwrite("out/nb1_bev_render.png", bev_true)
cv2.imwrite("out/nb1_bev_ipm.png", bev_ipm)
cv2.imwrite("out/nb1_bev.png", np.hstack([bev_true, gap, bev_ipm]))
print("wrote out/nb1_*.png  (nb1_bev.png: 왼쪽 정답 BEV | 오른쪽 IPM BEV)")
