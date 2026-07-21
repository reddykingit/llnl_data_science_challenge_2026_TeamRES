from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
MASK = ROOT / "data/9x9x9_octet_lattice/segmentation/segmented_mask.tif"
OUT = ROOT / ".codex/visualizations/2026/07/21/octet-lattice/octet-lattice-3d.html"


def main() -> None:
    # Sample the verified binary mask on a regular grid, then retain a bounded,
    # deterministic subset of occupied samples for smooth browser interaction.
    stride = 5
    chunks = []
    with tifffile.TiffFile(MASK) as tif:
        shape = (len(tif.pages), *tif.pages[0].shape)
        for z in range(0, shape[0], stride):
            yy, xx = np.nonzero(tif.pages[z].asarray()[::stride, ::stride])
            chunks.append(np.column_stack((xx, yy, np.full_like(xx, z // stride))))
    pts = np.concatenate(chunks).astype(np.float32)
    if len(pts) > 75_000:
        take = np.linspace(0, len(pts) - 1, 75_000, dtype=np.int64)
        pts = pts[take]

    dims = np.array([shape[2], shape[1], shape[0]], dtype=np.float32) / stride
    pts -= dims / 2
    pts /= dims.max()
    payload = base64.b64encode(pts.astype("<f4").tobytes()).decode("ascii")

    fragment = f'''<div id="octet-lattice-3d" style="width:100%;">
  <div class="viz-controls" aria-label="3D lattice controls">
    <label class="form-label" for="octet-point-size">Point size <span id="octet-point-value">1.8</span>
      <input class="form-range" id="octet-point-size" type="range" min="0.7" max="4" value="1.8" step="0.1">
    </label>
    <label class="form-label" for="octet-depth">Visible depth <span id="octet-depth-value">100%</span>
      <input class="form-range" id="octet-depth" type="range" min="5" max="100" value="100" step="1">
    </label>
    <button class="btn" id="octet-rotate" type="button" aria-pressed="true"><i data-lucide="rotate-3d" aria-hidden="true"></i> Auto-rotate</button>
    <button class="btn" id="octet-reset" type="button"><i data-lucide="maximize" aria-hidden="true"></i> Reset view</button>
  </div>
  <div id="octet-canvas" style="height:560px;min-height:360px;width:100%;position:relative;" role="img" aria-label="Interactive 3D point-cloud rendering of the segmented 9 by 9 by 9 octet lattice. Drag to orbit, scroll to zoom, and right-drag to pan."></div>
  <div class="viz-row text-small text-muted" aria-live="polite"><span>{len(pts):,} sampled foreground points</span><span>Drag: orbit</span><span>Wheel: zoom</span><span>Right-drag: pan</span></div>
</div>
<script type="module">
import * as THREE from 'https://esm.sh/three@0.164.1';
import {{ OrbitControls }} from 'https://esm.sh/three@0.164.1/examples/jsm/controls/OrbitControls.js';

const root = document.getElementById('octet-lattice-3d');
const host = document.getElementById('octet-canvas');
const raw = atob('{payload}');
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const positions = new Float32Array(bytes.buffer);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
camera.position.set(1.15, 0.9, 1.35);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.computeBoundingSphere();
const color = getComputedStyle(root).getPropertyValue('--viz-series-1').trim() || '#4f8cff';
const material = new THREE.PointsMaterial({{ color, size: 0.009, sizeAttenuation: true, transparent: true, opacity: 0.9 }});
const cloud = new THREE.Points(geometry, material);
cloud.rotation.x = -0.08;
scene.add(cloud);

const box = new THREE.Box3().setFromObject(cloud);
const helper = new THREE.Box3Helper(box, new THREE.Color(getComputedStyle(root).getPropertyValue('--border').trim() || '#888888'));
helper.material.transparent = true;
helper.material.opacity = 0.38;
scene.add(helper);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.55;
controls.maxDistance = 4;
let rotating = true;

const resetView = () => {{
  camera.position.set(1.15, 0.9, 1.35);
  controls.target.set(0, 0, 0);
  controls.update();
}};
resetView();

const pointSlider = document.getElementById('octet-point-size');
pointSlider.addEventListener('input', () => {{
  document.getElementById('octet-point-value').textContent = pointSlider.value;
  material.size = Number(pointSlider.value) * 0.005;
}});

const depthSlider = document.getElementById('octet-depth');
depthSlider.addEventListener('input', () => {{
  const pct = Number(depthSlider.value);
  document.getElementById('octet-depth-value').textContent = pct + '%';
  geometry.setDrawRange(0, Math.floor(geometry.attributes.position.count * pct / 100));
}});

const rotateButton = document.getElementById('octet-rotate');
rotateButton.addEventListener('click', () => {{
  rotating = !rotating;
  rotateButton.setAttribute('aria-pressed', String(rotating));
}});
document.getElementById('octet-reset').addEventListener('click', resetView);

function resize() {{
  const width = Math.max(host.clientWidth, 320);
  const height = Math.max(360, Math.min(560, width * 0.72));
  host.style.height = height + 'px';
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}}
new ResizeObserver(resize).observe(host);
resize();

let prior = performance.now();
function animate(now) {{
  const dt = Math.min((now - prior) / 1000, 0.05);
  prior = now;
  if (rotating && !matchMedia('(prefers-reduced-motion: reduce)').matches) cloud.rotation.y += dt * 0.18;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}}
requestAnimationFrame(animate);
</script>
'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(fragment, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(pts):,} points)")


if __name__ == "__main__":
    main()
