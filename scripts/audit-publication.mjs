import { lstat, readdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const fileLimit = 256 * 1024;
const reviewedLargeFiles = new Set([
  'bridge/static/vendor/three.module.js',
  'bridge/static/vendor/gaussian-splats-3d.module.js',
]);
const rules = [
  ['provider_key', /\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{20,}\b/],
  ['github_token', /\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b/],
  ['cloud_access_key', /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/],
  ['private_key', /-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----/],
  ['windows_personal_path', /\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s"']+/i],
  ['unix_personal_path', /\/(?:home|Users)\/[^\s/"']+/],
  ['credential_url', /https?:\/\/[^/\s:@]+:[^/\s@]+@/i],
];

// A narrow guard against common accidental disclosures, not a complete secret detector.
export function scanText(value) {
  return rules.filter(([, pattern]) => pattern.test(value)).map(([code]) => code);
}

export async function auditPublication() {
  const manifestPath = resolve(root, 'PUBLICATION_FILES.json');
  const manifestStat = await lstat(manifestPath);
  if (!manifestStat.isFile() || manifestStat.isSymbolicLink() || manifestStat.size > fileLimit) throw new Error('invalid_manifest');
  const files = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (!Array.isArray(files) || files.length === 0 || files.some(file =>
    typeof file !== 'string' || !/^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(file) ||
    file.split('/').some(part => ['.', '..', '.git'].includes(part)))) throw new Error('invalid_manifest');
  const expected = new Set(files);
  if (expected.size !== files.length || !expected.has('PUBLICATION_FILES.json')) throw new Error('invalid_manifest');
  const seen = new Set();
  const issues = [];
  async function visit(directory, prefix = '') {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (prefix === '' && entry.name === '.git') continue;
      // CI checks the clean checkout before installing dependencies.
      // Local environments/caches are not part of the source release.
      if (entry.isDirectory() && (entry.name === '__pycache__' || entry.name === '.pytest_cache' ||
          (prefix === '' && ['.venv', 'dist'].includes(entry.name)) ||
          (prefix === 'tools/splat-transform/' && entry.name === 'node_modules'))) continue;
      const name = prefix + entry.name;
      const path = resolve(directory, entry.name);
      const stat = await lstat(path);
      if (stat.isSymbolicLink()) { issues.push({ code: 'linked_entry' }); continue; }
      if (stat.isDirectory()) {
        if (!files.some(file => file.startsWith(name + '/'))) { issues.push({ code: 'unexpected_directory' }); continue; }
        await visit(path, name + '/');
        continue;
      }
      if (!stat.isFile() || !expected.has(name)) { issues.push({ code: 'unexpected_file' }); continue; }
      seen.add(name);
      if (stat.nlink > 1) { issues.push({ file: name, code: 'hard_linked_file' }); continue; }
      if (stat.size > (reviewedLargeFiles.has(name) ? 2 * 1024 * 1024 : fileLimit)) {
        issues.push({ file: name, code: 'file_too_large' }); continue;
      }
      const content = await readFile(path);
      if (content.includes(0)) { issues.push({ file: name, code: 'binary_content' }); continue; }
      for (const code of scanText(content.toString('utf8'))) issues.push({ file: name, code });
    }
  }
  await visit(root);
  for (const file of expected) if (!seen.has(file)) issues.push({ file, code: 'missing_file' });
  return { ok: issues.length === 0, checkedFiles: seen.size, issues };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = await auditPublication();
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
    if (!result.ok) process.exitCode = 1;
  } catch {
    process.stderr.write('Publication audit failed to read or validate the package.\n');
    process.exitCode = 2;
  }
}
