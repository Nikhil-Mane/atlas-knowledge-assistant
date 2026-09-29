# The Node.js Event Loop

Node.js runs JavaScript on a single thread. The event loop lets it handle
thousands of concurrent connections by offloading I/O to the operating system
and the libuv thread pool, then running callbacks when results are ready.

## The six phases

Each turn of the loop (a "tick") moves through these phases in order:

1. **timers**: runs callbacks scheduled by `setTimeout()` and `setInterval()`.
2. **pending callbacks**: runs I/O callbacks deferred from the previous tick.
3. **idle, prepare**: used internally by Node.js.
4. **poll**: retrieves new I/O events and runs their callbacks.
5. **check**: runs `setImmediate()` callbacks.
6. **close callbacks**: runs `close` event handlers, such as `socket.on('close')`.

## Microtasks run between phases

`process.nextTick()` callbacks run first, before any promise callbacks.
Resolved promise callbacks (`.then`, `await` continuations) run next. Both
queues drain completely before the loop moves on to the next phase.

```js
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));
Promise.resolve().then(() => console.log('promise'));
process.nextTick(() => console.log('nextTick'));
// Output: nextTick, promise, then timeout/immediate (order varies in main module)
```

## Do not block the loop

CPU-heavy work such as large `JSON.parse` calls, synchronous crypto or
`fs.readFileSync` in a request handler stalls every other request.
Move heavy work to `worker_threads` or split it into smaller chunks.

## The libuv thread pool

File system operations, DNS lookups (`dns.lookup`), `crypto.pbkdf2` and
`zlib` compression run on the libuv thread pool. It has **4 threads by default**.
Raise it with the `UV_THREADPOOL_SIZE` environment variable (maximum 1024).
