const http = require('http');
const fs = require('fs');
const path = require('path');

const baseDir = process.argv[2];
const port = Number(process.argv[3] || 8899);
if (!baseDir) {
  console.error('Usage: node simple_static_server.js <baseDir> [port]');
  process.exit(2);
}

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.glb': 'model/gltf-binary',
  '.stl': 'model/stl',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
};

const normalizedBase = path.normalize(baseDir);
const server = http.createServer((req, res) => {
  try {
    let reqPath = decodeURIComponent((req.url || '/').split('?')[0]);
    if (reqPath === '/') reqPath = '/winding_workbench/model_demo/index.html';
    let filePath = path.normalize(path.join(normalizedBase, reqPath));

    if (!filePath.startsWith(normalizedBase)) {
      res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Forbidden');
      return;
    }

    if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
      filePath = path.join(filePath, 'index.html');
    }

    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': mime[ext] || 'application/octet-stream' });
    fs.createReadStream(filePath).pipe(res);
  } catch (error) {
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(String(error));
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Static server ready: http://127.0.0.1:${port}`);
  console.log(`Base dir: ${normalizedBase}`);
});
