/**
 * Minimal production-style Express server.
 * Shows: JSON body limit, request validation, async error handling,
 * a health check, and graceful shutdown on SIGTERM.
 */
const express = require('express');

const app = express();
const PORT = process.env.PORT || 3000;

// Reject bodies larger than 100 KB to protect memory.
app.use(express.json({ limit: '100kb' }));

const users = new Map();

// Health check used by Docker / Kubernetes liveness probes.
app.get('/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

app.post('/users', async (req, res, next) => {
  try {
    const { name, email } = req.body;
    if (!name || !email) {
      return res.status(400).json({ error: 'name and email are required' });
    }
    const id = crypto.randomUUID();
    users.set(id, { id, name, email });
    return res.status(201).json(users.get(id));
  } catch (err) {
    return next(err); // Forward to the error middleware below.
  }
});

app.get('/users/:id', (req, res) => {
  const user = users.get(req.params.id);
  if (!user) return res.status(404).json({ error: 'user not found' });
  return res.json(user);
});

// Error-handling middleware must take exactly four arguments.
app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: 'internal server error' });
});

const server = app.listen(PORT, () => {
  console.log(`Listening on http://localhost:${PORT}`);
});

// Graceful shutdown: stop accepting connections, finish in-flight requests.
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing server');
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 10_000).unref(); // Force exit after 10 s.
});
