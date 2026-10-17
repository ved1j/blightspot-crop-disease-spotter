// Offline app shell: cache static assets; always go to the network for API calls.
const CACHE = "blightspot-shell-v2";
const SHELL = ["./", "index.html", "manifest.webmanifest", "icons/favicon.svg",
               "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png"];
const API = ["/analyze", "/sample", "/classes", "/metrics", "/health"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  if (API.some((p) => url.pathname.endsWith(p))) return;
  // Network-first so deploys show up immediately; fall back to cache when offline.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(() => caches.match(e.request).then((r) => r || caches.match("index.html")))
  );
});
