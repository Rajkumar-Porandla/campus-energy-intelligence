/**
 * Script to drive headless Google Chrome via Chrome DevTools Protocol (CDP)
 * using Node.js built-in fetch and WebSocket.
 * 
 * Captures full-page desktop (1440x1080) and mobile (390x844) screenshots,
 * measures control alignment in Consumption Analytics, and checks for console/DOM errors.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9222;
const APP_URL = "http://localhost:8501";
const SCREENSHOTS_DIR = path.resolve("docs/screenshots");

// Ensure screenshots dir exists
if (!fs.existsSync(SCREENSHOTS_DIR)) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 1. Launch Chrome
console.log("Launching headless Google Chrome...");
const chromeProcess = spawn(
  CHROME_PATH,
  [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    "--window-size=1440,1080",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    APP_URL,
  ],
  { stdio: "ignore" }
);

// Clean up on exit
process.on("exit", () => {
  try {
    chromeProcess.kill("SIGKILL");
  } catch (e) {}
});

await sleep(2500);

// 2. Query CDP targets
const targetsRes = await fetch(`http://127.0.0.1:${PORT}/json`);
const targets = await targetsRes.json();
const pageTarget = targets.find((t) => t.type === "page") || targets[0];

if (!pageTarget || !pageTarget.webSocketDebuggerUrl) {
  console.error("Failed to find Chrome page target. Targets:", targets);
  process.exit(1);
}

console.log(`Connecting to CDP target: ${pageTarget.title} (${pageTarget.webSocketDebuggerUrl})`);

const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);

await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});

let msgId = 1;
const pending = new Map();
const consoleErrors = [];

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
    consoleErrors.push(msg.params.args.map((a) => a.value || a.description).join(" "));
  }
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) {
      reject(new Error(msg.error.message));
    } else {
      resolve(msg.result);
    }
  }
};

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = msgId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
}

// Enable domains
await send("Page.enable");
await send("Runtime.enable");

console.log("Waiting for Streamlit initial render...");
await sleep(4000);

const sections = [
  { name: "page_1_command_center", label: "1. Command Center" },
  { name: "page_2_consumption_analytics", label: "2. Consumption Analytics" },
  { name: "page_3_forecasting_lab", label: "3. Forecasting Lab" },
  { name: "page_4_anomaly_observatory", label: "4. Anomaly Observatory" },
  { name: "page_5_building_intelligence", label: "5. Building Intelligence" },
  { name: "page_6_energy_insights", label: "6. Energy Insights" },
];

// Set desktop viewport initially
await send("Emulation.setDeviceMetricsOverride", {
  width: 1440,
  height: 1080,
  deviceScaleFactor: 1,
  mobile: false,
});

for (const section of sections) {
  console.log(`Navigating to section: ${section.label}...`);
  // Click on radio button with corresponding label
  const clickScript = `
    (() => {
      const radios = Array.from(document.querySelectorAll('div[data-testid="stRadio"] label'));
      const target = radios.find(el => el.textContent && el.textContent.includes("${section.label}"));
      if (target) {
        target.click();
        return { success: true, text: target.textContent.trim() };
      }
      return { success: false, available: radios.map(r => r.textContent.trim()) };
    })()
  `;

  const evalRes = await send("Runtime.evaluate", {
    expression: clickScript,
    returnByValue: true,
  });

  const clickResult = evalRes.result.value;
  console.log("Click result:", clickResult);

  // Wait for content and Plotly charts to render
  await sleep(3500);

  // If on Consumption Analytics, measure control positions
  if (section.name === "page_2_consumption_analytics") {
    const alignmentCheck = await send("Runtime.evaluate", {
      expression: `
        (() => {
          const selectboxes = Array.from(document.querySelectorAll('div[data-testid="stSelectbox"]')).map(el => {
            const rect = el.getBoundingClientRect();
            const labelEl = el.querySelector('label');
            return {
              label: labelEl ? labelEl.textContent.trim() : 'Unknown',
              top: Math.round(rect.top),
              left: Math.round(rect.left),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            };
          });

          const checkboxes = Array.from(document.querySelectorAll('div[data-testid="stCheckbox"]')).map(el => {
            const rect = el.getBoundingClientRect();
            return {
              text: el.textContent.trim(),
              top: Math.round(rect.top),
              left: Math.round(rect.left),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            };
          });

          return { selectboxes, checkboxes };
        })()
      `,
      returnByValue: true,
    });
    console.log("=== CONSUMPTION ANALYTICS CONTROL ALIGNMENT AUDIT ===");
    console.log(JSON.stringify(alignmentCheck.result.value, null, 2));
  }

  // Capture desktop screenshot
  const screenshotRes = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });

  const imgBuffer = Buffer.from(screenshotRes.data, "base64");
  const filePath = path.join(SCREENSHOTS_DIR, `${section.name}.png`);
  fs.writeFileSync(filePath, imgBuffer);
  console.log(`Saved desktop screenshot: ${filePath} (${(imgBuffer.length / 1024).toFixed(1)} KB)`);
}

// 3. Mobile Viewport Verification (iPhone 14 / Mobile 390x844)
console.log("Switching to Mobile Viewport (390x844)...");
await send("Emulation.setDeviceMetricsOverride", {
  width: 390,
  height: 844,
  deviceScaleFactor: 2,
  mobile: true,
});
await sleep(2500);

// Capture mobile screenshot of Consumption Analytics
const mobileRes = await send("Page.captureScreenshot", {
  format: "png",
  captureBeyondViewport: false,
});
const mobileBuffer = Buffer.from(mobileRes.data, "base64");
const mobilePath = path.join(SCREENSHOTS_DIR, "mobile_page_2_consumption_analytics.png");
fs.writeFileSync(mobilePath, mobileBuffer);
console.log(`Saved mobile screenshot: ${mobilePath} (${(mobileBuffer.length / 1024).toFixed(1)} KB)`);

// Check for unhandled exceptions or error messages in the DOM
const errorCheck = await send("Runtime.evaluate", {
  expression: `
    (() => {
      const alerts = Array.from(document.querySelectorAll('[data-testid="stException"], [data-testid="stAlert"]'))
        .map(el => el.textContent.trim());
      return alerts;
    })()
  `,
  returnByValue: true,
});

console.log("DOM Exception/Alert check:", errorCheck.result.value);
console.log("Captured Console Errors:", consoleErrors);

ws.close();
chromeProcess.kill("SIGKILL");
console.log("Browser inspection complete!");
