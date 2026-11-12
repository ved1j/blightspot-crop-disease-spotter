// Offline support: network-first for our own files (so deploys show up at once),
// cache-first for the pinned onnxruntime-web CDN files, never cache API calls.
const CACHE = "blightspot-v3";
const SHELL = ["./", "index.html", "engine.js", "manifest.webmanifest", "model/meta.json",
               "icons/favicon.svg", "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png"];
const API = ["/analyze", "/sample", "/classes", "/metrics", "/health"];
const CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@";

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  if (req.url.startsWith(CDN)) { // versioned + immutable: cache-first
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    })));
    return;
  }
  if (url.origin !== location.origin || API.some((p) => url.pathname.endsWith(p))) return;

  e.respondWith(
    fetch(req)
      .then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
        return res;
      })
      .catch(() => caches.match(req).then((r) => r || caches.match("index.html")))
  );
});
