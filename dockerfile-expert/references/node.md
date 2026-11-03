# Node.js and TypeScript images

## Default: npm (assets/Dockerfile.node)

Stages:
- **base**: `node:<LTS>-trixie-slim`, `WORKDIR /app`.
- **deps**: `npm ci --omit=dev` with package.json and package-lock.json bind-mounted and
  `/root/.npm` as a cache mount. Produces production `node_modules` only.
- **build**: full `npm ci` (dev dependencies for TypeScript, bundlers), then source, then
  `npm run build`.
- **runtime**: `NODE_ENV=production`, copies prod `node_modules`, the build output, and
  package.json, runs as UID 1000 (the image's `node` user).

Tested result (October 2026, Engine 29.9, buildx 0.38, architecture not recorded; stop times vary by host): served healthz as UID 1000 and stopped in 0.2 s, because the test server
registers a SIGTERM handler (see below).

Use an LTS major for production. Check nodejs.org's release table: on 19 October 2026, 24 was
LTS and 26 was Current (even-numbered lines have historically entered LTS each October; from
Node 27 the project moves to one major per year, each becoming LTS).

## Signals: the most common Node container bug

Node as PID 1 ignores SIGTERM unless the app handles it (tested October 2026, architecture not recorded: a plain `node -e` loop was
SIGKILLed after 10 s). Either add a handler that closes the server:

```ts
const server = app.listen(port);
const shutdown = () => {
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 10_000).unref();
};
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
```

or run under an init (`docker run --init`, or install `tini` and use
`ENTRYPOINT ["tini", "--"]`). Never `CMD ["npm", "start"]`: npm adds a process between the
signal and your server.

## pnpm and Yarn

pnpm:
```dockerfile
RUN npm install -g pnpm@<pinned-version>
RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=pnpm-lock.yaml,target=pnpm-lock.yaml \
    --mount=type=cache,target=/pnpm/store \
    pnpm install --frozen-lockfile --prod --store-dir /pnpm/store
```
(Corepack can also supply pnpm or Yarn, but whether it ships with Node depends on the Node
version; check before relying on it.)

Yarn Berry: `yarn install --immutable`. With Plug'n'Play, copy `.pnp.cjs` and `.yarn/` into the
runtime stage; with `nodeLinker: node-modules` it behaves like npm.

For monorepos, `pnpm deploy --prod <dir>` or `turbo prune` produce a pruned copy of one
package with only its dependencies, which then becomes the runtime stage's input.

## Frameworks

- **Next.js**: set `output: "standalone"` in next.config; copy `.next/standalone`,
  `.next/static` (to `.next/static`) and `public` into the runtime stage and run
  `node server.js`. The standalone output already includes the needed `node_modules`.
- **Static SPAs** (Vite, CRA): build in Node, ship the `dist/` on an unprivileged web server
  image (`nginxinc/nginx-unprivileged` listens on 8080 as non-root) or a CDN; no Node in the
  final image.

## Native modules

Modules with node-gyp builds (bcrypt, sharp, better-sqlite3) need `python3 make g++` in the
**build/deps** stage only. Build and run on the same libc: a module compiled on Alpine (musl)
fails on Debian (glibc) and the reverse. Do `npm ci` in a stage based on the same image as the
runtime.

## Read-only root filesystem

Node itself does not need a writable root; libraries that write caches (some image libraries,
`npm` itself) do. Mount `emptyDir` at `/tmp` and point caches there.
