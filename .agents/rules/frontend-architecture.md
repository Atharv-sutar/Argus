# Frontend & UI Architecture Rules

## 1. Zero External Framework Dependencies
- The web frontend in `src/multi_camera/static/` is intentionally built with vanilla JavaScript (ES6+), standard CSS3 with CSS variables, and HTML5 standard APIs.
- Do not introduce React, Vue, Webpack, Vite, or external npm package build steps unless explicitly requested.

## 2. DOM Integrity & Element IDs
- Every element ID in `index.html` must be globally unique across all views and modals. Never duplicate IDs between the Live Matrix and Topology Map views.
- Use data attributes (`data-camera-id`, `data-layout`) and CSS classes for repeating collections (e.g. camera cards, gallery crops, list rows).

## 3. High-Throughput Streaming & Telemetry
- Telemetry SSE stream (`/api/stream/events`) transmits updates at ~10Hz. Client update handlers must be lightweight to prevent browser main-thread frame drops.
- Video feeds are delivered via MJPEG streams (`/api/camera/<id>/stream`). Camera tiles not currently active or visible should be throttled or rendered on-demand to conserve bandwidth and CPU decode overhead.
- Canvas renderings and export utilities (e.g. flowchart exporter, topology canvas) must use high DPI scaling and avoid blocking user interaction.

