function calcWater(heights) {
  const n = heights.length;
  const leftMax = [];
  const rightMax = [];

  leftMax[0] = heights[0];
  for (let i = 1; i < n; i++) {
    leftMax[i] = Math.max(leftMax[i - 1], heights[i]);
  }

  rightMax[n - 1] = heights[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    rightMax[i] = Math.max(rightMax[i + 1], heights[i]);
  }

  const water = heights.map((h, i) => Math.min(leftMax[i], rightMax[i]) - h);
  const total = water.reduce((sum, w) => sum + w, 0);
  return { water, total };
}

function parseInput(text) {
  const parts = text.split(",").map((s) => s.trim());
  if (parts.some((p) => p === "")) return null;
  const nums = parts.map(Number);
  if (nums.some((x) => !Number.isInteger(x) || x < 0)) return null;
  return nums;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const CELL = 40;

function drawTank(heights, water) {
  const svg = document.getElementById("tank");
  svg.innerHTML = "";
  const maxH = Math.max(...heights, 1);
  svg.setAttribute("viewBox", `0 0 ${heights.length * CELL} ${maxH * CELL}`);

  heights.forEach((h, i) => {
    for (let level = 0; level < maxH; level++) {
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", i * CELL);
      rect.setAttribute("y", (maxH - 1 - level) * CELL);
      rect.setAttribute("width", CELL);
      rect.setAttribute("height", CELL);
      rect.setAttribute("stroke", "#999");

      if (level < h) rect.setAttribute("fill", "#ffeb00");
      else if (level < h + water[i]) rect.setAttribute("fill", "#00b0f0");
      else rect.setAttribute("fill", "#fff");

      svg.appendChild(rect);
    }
  });
}

function run() {
  const error = document.getElementById("error");
  const result = document.getElementById("result");
  const heights = parseInput(document.getElementById("input").value);

  if (!heights) {
    error.textContent = "Enter whole numbers (0 or more) separated by commas.";
    result.textContent = "";
    document.getElementById("tank").innerHTML = "";
    return;
  }
  error.textContent = "";
  const { water, total } = calcWater(heights);
  result.textContent = `Output: ${total} Units`;
  drawTank(heights, water);
}

document.getElementById("run").addEventListener("click", run);
run();