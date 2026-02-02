/**
 * Cloudflare Worker serving a PEP 503/691 compliant Python package index from R2.
 * Supports both HTML (v1) and JSON responses per Simple Repository API spec.
 *
 * Routes:
 *   GET /simple/           - Package index listing
 *   GET /simple/{pkg}/     - Package version listing
 *   GET /packages/{file}   - Download wheel/sdist
 */

interface Env {
  BUCKET: R2Bucket;
  AUTH_TOKEN: string;
}

interface FileInfo {
  filename: string;
  url: string;
  hashes: { sha256?: string };
  size?: number;
  "requires-python"?: string;
}

interface ProjectListResponse {
  meta: { "api-version": string };
  projects: Array<{ name: string }>;
}

interface ProjectDetailResponse {
  meta: { "api-version": string };
  name: string;
  files: FileInfo[];
}

const API_VERSION = "1.1";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const wantsJson = acceptsJson(request);

    // Health check (no auth required)
    if (path === "/" || path === "/health") {
      return new Response("OK", { status: 200 });
    }

    // Check authentication
    if (!isAuthenticated(request, env.AUTH_TOKEN)) {
      return new Response("Unauthorized", {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="pypi"' },
      });
    }

    // PEP 503/691: Simple API root
    if (path === "/simple" || path === "/simple/") {
      return await listPackages(env.BUCKET, url.origin, wantsJson);
    }

    // PEP 503/691: Package detail page
    const packageMatch = path.match(/^\/simple\/([^/]+)\/?$/);
    if (packageMatch) {
      const packageName = normalizePackageName(packageMatch[1]);
      return await listPackageVersions(env.BUCKET, packageName, url.origin, wantsJson);
    }

    // Download package file
    const downloadMatch = path.match(/^\/packages\/(.+)$/);
    if (downloadMatch) {
      const filename = downloadMatch[1];
      return await downloadPackage(env.BUCKET, filename);
    }

    return new Response("Not Found", { status: 404 });
  },
};

/**
 * Check if client prefers JSON response
 */
function acceptsJson(request: Request): boolean {
  const accept = request.headers.get("Accept") || "";
  // Check for JSON content type per PEP 691
  if (accept.includes("application/vnd.pypi.simple.v1+json")) return true;
  if (accept.includes("application/json")) return true;
  return false;
}

/**
 * Check if request has valid authentication
 * Supports HTTP Basic Auth with token as username (password ignored)
 */
function isAuthenticated(request: Request, token: string): boolean {
  const auth = request.headers.get("Authorization");
  if (!auth?.startsWith("Basic ")) return false;

  try {
    const decoded = atob(auth.slice(6));
    const username = decoded.split(":")[0];
    return username === token;
  } catch {
    return false;
  }
}

/**
 * Normalize package name per PEP 503: lowercase, replace [-_.] with -
 */
function normalizePackageName(name: string): string {
  return name.toLowerCase().replace(/[-_.]+/g, "-");
}

/**
 * List all packages in the index
 */
async function listPackages(bucket: R2Bucket, origin: string, asJson: boolean): Promise<Response> {
  const packages = new Set<string>();

  // List all objects in packages/ prefix to find unique package names
  let cursor: string | undefined;
  do {
    const list = await bucket.list({ prefix: "packages/", cursor });
    for (const obj of list.objects) {
      const filename = obj.key.replace("packages/", "");
      // Extract package name from wheel filename: {name}-{version}...
      const match = filename.match(/^([^-]+(?:-[^-]+)*?)-\d/);
      if (match) {
        packages.add(normalizePackageName(match[1]));
      }
    }
    cursor = list.truncated ? list.cursor : undefined;
  } while (cursor);

  const sortedPackages = Array.from(packages).sort();

  if (asJson) {
    const response: ProjectListResponse = {
      meta: { "api-version": API_VERSION },
      projects: sortedPackages.map(name => ({ name })),
    };
    return new Response(JSON.stringify(response), {
      headers: {
        "Content-Type": "application/vnd.pypi.simple.v1+json",
      },
    });
  }

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta name="pypi:repository-version" content="${API_VERSION}">
  <title>Simple Index</title>
</head>
<body>
<h1>Simple Index</h1>
${sortedPackages.map(pkg => `<a href="/simple/${pkg}/">${pkg}</a><br/>`).join("\n")}
</body>
</html>`;

  return new Response(html, {
    headers: { "Content-Type": "application/vnd.pypi.simple.v1+html; charset=utf-8" },
  });
}

/**
 * List all versions of a specific package
 */
async function listPackageVersions(
  bucket: R2Bucket,
  packageName: string,
  origin: string,
  asJson: boolean
): Promise<Response> {
  const files: FileInfo[] = [];

  let cursor: string | undefined;
  do {
    const list = await bucket.list({ prefix: "packages/", cursor });
    for (const obj of list.objects) {
      const filename = obj.key.replace("packages/", "");
      // Check if this file belongs to the requested package
      const match = filename.match(/^([^-]+(?:-[^-]+)*?)-\d/);
      if (match && normalizePackageName(match[1]) === packageName) {
        const sha256 = obj.customMetadata?.sha256;
        files.push({
          filename,
          url: `${origin}/packages/${filename}`,
          hashes: sha256 ? { sha256 } : {},
          size: obj.size,
        });
      }
    }
    cursor = list.truncated ? list.cursor : undefined;
  } while (cursor);

  if (files.length === 0) {
    return new Response("Package Not Found", { status: 404 });
  }

  if (asJson) {
    const response: ProjectDetailResponse = {
      meta: { "api-version": API_VERSION },
      name: packageName,
      files,
    };
    return new Response(JSON.stringify(response), {
      headers: {
        "Content-Type": "application/vnd.pypi.simple.v1+json",
      },
    });
  }

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta name="pypi:repository-version" content="${API_VERSION}">
  <title>Links for ${packageName}</title>
</head>
<body>
<h1>Links for ${packageName}</h1>
${files.map(f => {
  const hashAttr = f.hashes.sha256 ? ` data-dist-info-metadata="sha256=${f.hashes.sha256}"` : "";
  return `<a href="${f.url}"${hashAttr}>${f.filename}</a><br/>`;
}).join("\n")}
</body>
</html>`;

  return new Response(html, {
    headers: { "Content-Type": "application/vnd.pypi.simple.v1+html; charset=utf-8" },
  });
}

/**
 * Download a package file from R2
 */
async function downloadPackage(bucket: R2Bucket, filename: string): Promise<Response> {
  const obj = await bucket.get(`packages/${filename}`);

  if (!obj) {
    return new Response("File Not Found", { status: 404 });
  }

  const headers = new Headers();
  headers.set("Content-Type", "application/octet-stream");
  headers.set("Content-Disposition", `attachment; filename="${filename}"`);
  if (obj.size) {
    headers.set("Content-Length", obj.size.toString());
  }
  if (obj.customMetadata?.sha256) {
    headers.set("X-Checksum-Sha256", obj.customMetadata.sha256);
  }

  return new Response(obj.body, { headers });
}
